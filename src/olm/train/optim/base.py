# src/olm/train/optim/base.py
import torch
from torch.optim.optimizer import Optimizer
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict


class OptimizerBase(Optimizer, ABC):
    """
    Abstract base class for all optimizers in the OLM framework.

    Provides a consistent interface for optimizer implementations, including
    standard methods for parameter updates, gradient zeroing, and state management.
    All custom optimizers should inherit from this class.

    This base class extends PyTorch's Optimizer class and adds additional
    functionality specific to the OLM framework.

    Subclasses must implement the step() method to define the optimization logic.
    """

    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Some optimization algorithms (e.g., L-BFGS) require multiple
                evaluations of the loss function.

        Returns:
            Optional loss value if closure is provided.
        """
        pass

    def zero_grad(self, set_to_none: bool = True):
        """
        Sets gradients of all optimized tensors to zero or None.

        Args:
            set_to_none: Instead of setting to zero, set the grads to None.
                This is more memory efficient and can slightly improve performance.
                Default: True
        """
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a dict.

        It contains two entries:

        - ``state``: dict holding current optimization state. Its content
          differs between optimizer classes.
        - ``param_groups``: list containing all parameter groups where each
          parameter group is a dict.

        Returns:
            Dictionary containing optimizer state
        """
        return super().state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the optimizer state.

        Args:
            state_dict: optimizer state. Should be an object returned
                from a call to state_dict().
        """
        super().load_state_dict(state_dict)

    def extra_repr(self) -> str:
        """
        String representation of the optimizer for debugging.

        Override this in subclasses to provide useful information.
        """
        return ""
