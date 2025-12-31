"""Open Language Model (OLM) - A clean PyTorch library for LLM development."""

__version__ = "0.1.0"

# Core modules
from olm import core
from olm import nn
from olm import models
from olm import data
from olm import train
from olm import logging
from olm import export

# Common imports for convenience
from olm.models import GPT, LLaMA
from olm.train.trainer import Trainer
from olm.train.callbacks import (
    CheckpointCallback,
    ValidationCallback,
    MetricsLoggerCallback,
    ThroughputCallback,
    LRMonitorCallback,
    EarlyStoppingCallback,
)
from olm.train.schedulers import (
    SchedulerBase,
    CosineAnnealingLR,
    LinearLR,
    WarmupLR,
    WarmupCosineScheduler,
)
from olm.train.optim import OptimizerBase, AdamW, Lion
from olm.data.datasets import DataLoader

__all__ = [
    "__version__",
    # Modules
    "core",
    "nn",
    "models",
    "data",
    "train",
    "logging",
    "export",
    # Models
    "GPT",
    "LLaMA",
    # Training
    "Trainer",
    "CheckpointCallback",
    "ValidationCallback",
    "MetricsLoggerCallback",
    "ThroughputCallback",
    "LRMonitorCallback",
    "EarlyStoppingCallback",
    # Schedulers
    "SchedulerBase",
    "CosineAnnealingLR",
    "LinearLR",
    "WarmupLR",
    "WarmupCosineScheduler",
    # Optimizers
    "OptimizerBase",
    "AdamW",
    "Lion",
    # Data
    "DataLoader",
]
