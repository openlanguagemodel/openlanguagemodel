"""Training infrastructure for OLM."""

from olm.train.trainer import Trainer, TrainerCallback
from olm.train import callbacks
from olm.train import optim
from olm.train import schedulers
from olm.train import losses
from olm.train import regularization

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

__all__ = [
    # Core
    "Trainer",
    "TrainerCallback",
    # Submodules
    "callbacks",
    "optim",
    "schedulers",
    "losses",
    "regularization",
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
]
