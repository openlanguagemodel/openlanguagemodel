"""Logging utilities for OLM."""

from olm.logging.logger import Logger
from olm.logging.progress import ProgressBar

# WandB integration (optional dependency)
try:
    from olm.logging.wandb_logger import (
        WandBCallback,
        create_sweep,
        get_sweep_config_template,
    )

    __all__ = [
        "Logger",
        "ProgressBar",
        "WandBCallback",
        "create_sweep",
        "get_sweep_config_template",
    ]
except ImportError:
    __all__ = ["Logger", "ProgressBar"]
