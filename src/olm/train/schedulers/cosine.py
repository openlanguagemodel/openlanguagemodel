"""Cosine annealing learning rate scheduler."""

import math
from olm.train.schedulers.base import SchedulerBase


class CosineAnnealingLR(SchedulerBase):
    """
    Cosine annealing learning rate scheduler.

    Decreases the learning rate following a cosine curve from the initial
    learning rate to eta_min over T_max steps.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of iterations (steps).
        eta_min: Minimum learning rate (default: 0).
        last_epoch: The index of last epoch (default: -1).

    Example:
        >>> from olm.train.schedulers import CosineAnnealingLR
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        >>> for epoch in range(epochs):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using cosine annealing."""
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.T_max:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]
        else:
            return [self.eta_min for _ in self.base_lrs]

    def _get_closed_form_lr(self):
        """Closed form solution for learning rate."""
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
