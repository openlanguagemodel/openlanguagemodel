"""Base learning rate scheduler for OLM."""

from abc import ABC, abstractmethod
from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class SchedulerBase(_LRScheduler, ABC):
    """
    Base class for all OLM learning rate schedulers.

    This class extends PyTorch's _LRScheduler and provides a consistent
    interface for implementing custom learning rate schedules. All OLM
    schedulers should inherit from this class to maintain uniformity.

    Subclasses must implement:
        - get_lr(): Compute the learning rate for the current step
        - _get_closed_form_lr() (optional): Closed-form solution for efficiency

    Args:
        optimizer: Wrapped PyTorch optimizer.
        last_epoch: The index of the last epoch (default: -1).
        verbose: If True, prints a message to stdout for each update (default: False).

    Example:
        >>> class MyScheduler(SchedulerBase):
        ...     def __init__(self, optimizer, param, last_epoch=-1):
        ...         self.param = param
        ...         super().__init__(optimizer, last_epoch)
        ...
        ...     def get_lr(self):
        ...         # Custom logic here
        ...         return [base_lr * self.param for base_lr in self.base_lrs]
    """

    def __init__(self, optimizer, last_epoch: int = -1, verbose: bool = False):
        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self) -> List[float]:
        """
        Compute learning rate for each parameter group.

        This method must be implemented by subclasses to define the
        learning rate schedule logic.

        Returns:
            List of learning rates, one per parameter group.
        """
        pass

    def _get_closed_form_lr(self) -> List[float]:
        """
        Optional closed-form solution for learning rate computation.

        Override this method if a closed-form solution is available
        for better performance.

        Returns:
            List of learning rates, one per parameter group.
        """
        return self.get_lr()

    def get_last_lr(self) -> List[float]:
        """
        Return last computed learning rate by current scheduler.

        Returns:
            List of last computed learning rates.
        """
        return self._last_lr

    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.

        Contains all non-callable attributes that are specific to
        the scheduler and required for checkpointing.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_get_lr_called_within_step")
        }

    def load_state_dict(self, state_dict):
        """
        Load the scheduler state from a checkpoint.

        Args:
            state_dict: Scheduler state returned by state_dict().
        """
        self.__dict__.update(state_dict)
