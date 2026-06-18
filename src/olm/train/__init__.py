"""Training infrastructure for OLM."""

from olm.train.trainer import (
    Trainer,
    TrainerCallback,
    DDPTrainer,
    FSDPTrainer,
    AutoTrainer,
    auto_trainer,
)
from olm.train import callbacks
from olm.train import optim
from olm.train import schedulers
from olm.train import losses
from olm.train import regularization
from olm.train import device

# Re-export common components
from olm.train.callbacks import (
    CheckpointCallback,
    ValidationCallback,
    MetricsLoggerCallback,
    ThroughputCallback,
    LRMonitorCallback,
    EarlyStoppingCallback,
)
from olm.train.optim import OptimizerBase, AdamW, Lion
from olm.train.schedulers import (
    SchedulerBase,
    CosineAnnealingLR,
    LinearLR,
    LinearDecayLR,
    WarmupLR,
    WarmupCosineScheduler,
)
from olm.train.device import (
    DeviceConfig,
    TrainerStrategy,
    detect_devices,
    determine_strategy,
    parse_device_string,
    estimate_model_size,
    print_strategy_summary,
)

__all__ = [
    # Core
    "Trainer",
    "TrainerCallback",
    "DDPTrainer",
    "FSDPTrainer",
    "AutoTrainer",
    "auto_trainer",
    # Submodules
    "callbacks",
    "optim",
    "schedulers",
    "losses",
    "regularization",
    "device",
    # Callbacks
    "CheckpointCallback",
    "ValidationCallback",
    "MetricsLoggerCallback",
    "ThroughputCallback",
    "LRMonitorCallback",
    "EarlyStoppingCallback",
    # Optimizers
    "OptimizerBase",
    "AdamW",
    "Lion",
    # Schedulers
    "SchedulerBase",
    "CosineAnnealingLR",
    "LinearLR",
    "LinearDecayLR",
    "WarmupLR",
    "WarmupCosineScheduler",
    # Device utilities
    "DeviceConfig",
    "TrainerStrategy",
    "detect_devices",
    "determine_strategy",
    "parse_device_string",
    "estimate_model_size",
    "print_strategy_summary",
]
