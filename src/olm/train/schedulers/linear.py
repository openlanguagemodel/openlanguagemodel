"""Linear learning rate scheduler."""

from olm.train.schedulers.base import SchedulerBase


class LinearLR(SchedulerBase):
    """
    Linear learning rate scheduler.

    Linearly decreases (or increases) the learning rate from the initial
    learning rate to end_lr over total_steps.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of steps for the schedule.
        end_lr: Target learning rate at the end (default: 0).
        start_factor: Initial learning rate multiplier (default: 1.0).
        last_epoch: The index of last epoch (default: -1).

    Example:
        >>> from olm.train.schedulers import LinearLR
        >>> # Decay from initial LR to 0
        >>> scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
        >>> for step in range(total_steps):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        end_lr: float = 0,
        start_factor: float = 1.0,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using linear interpolation."""
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]

        if self.last_epoch > self.total_steps:
            return [self.end_lr for _ in self.base_lrs]

        # Linear interpolation
        alpha = self.last_epoch / self.total_steps
        return [
            base_lr * self.start_factor * (1 - alpha) + self.end_lr * alpha
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        """Closed form solution for learning rate."""
        alpha = min(self.last_epoch / self.total_steps, 1.0)
        return [
            base_lr * self.start_factor * (1 - alpha) + self.end_lr * alpha
            for base_lr in self.base_lrs
        ]


class LinearDecayLR(SchedulerBase):
    """
    Simple linear decay scheduler that decays to zero.

    This is a simplified version that always decays to 0 from the initial LR.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of steps to decay over.
        last_epoch: The index of last epoch (default: -1).

    Example:
        >>> from olm.train.schedulers import LinearDecayLR
        >>> scheduler = LinearDecayLR(optimizer, total_steps=1000)
        >>> for step in range(total_steps):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer, total_steps: int, last_epoch: int = -1):
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using linear decay."""
        if self.last_epoch >= self.total_steps:
            return [0 for _ in self.base_lrs]

        decay_factor = 1 - (self.last_epoch / self.total_steps)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
