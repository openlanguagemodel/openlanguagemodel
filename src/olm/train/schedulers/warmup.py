"""Warmup learning rate scheduler."""

from olm.train.schedulers.base import SchedulerBase


class WarmupLR(SchedulerBase):
    """
    Learning rate warmup scheduler.

    Linearly increases the learning rate from 0 to the base learning rate
    over warmup_steps.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        start_lr: Initial learning rate (default: 0).
        last_epoch: The index of last epoch (default: -1).

    Example:
        >>> from olm.train.schedulers import WarmupLR
        >>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
        >>> for step in range(warmup_steps):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        start_lr: float = 0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate during warmup."""
        if self.last_epoch >= self.warmup_steps:
            return [base_lr for base_lr in self.base_lrs]

        # Linear warmup
        warmup_factor = (self.last_epoch + 1) / self.warmup_steps
        return [
            self.start_lr + (base_lr - self.start_lr) * warmup_factor
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        """Closed form solution for learning rate."""
        if self.last_epoch >= self.warmup_steps:
            return self.base_lrs

        warmup_factor = (self.last_epoch + 1) / self.warmup_steps
        return [
            self.start_lr + (base_lr - self.start_lr) * warmup_factor
            for base_lr in self.base_lrs
        ]


class WarmupCosineScheduler(SchedulerBase):
    """
    Combined warmup and cosine annealing scheduler.

    Linearly warms up the learning rate from 0 to base_lr over warmup_steps,
    then applies cosine annealing decay to min_lr over the remaining steps.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay (default: 0).
        last_epoch: The index of last epoch (default: -1).

    Example:
        >>> from olm.train.schedulers import WarmupCosineScheduler
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     min_lr=1e-6
        ... )
        >>> for step in range(total_steps):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate with warmup and cosine decay."""
        import math

        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing phase
        progress = (self.last_epoch - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)

        return [
            self.min_lr
            + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
