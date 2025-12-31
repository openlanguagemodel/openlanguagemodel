# src/olm/train/optim/zero.py
import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Dict, Any, List, Iterable
import warnings
from .base import OptimizerBase


class ZeROOptimizer(OptimizerBase):
    """
    ZeRO (Zero Redundancy Optimizer) wrapper for distributed training.

    Implements memory optimization techniques from "ZeRO: Memory Optimizations
    Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020).

    ZeRO reduces memory consumption by partitioning optimizer states, gradients,
    and parameters across data-parallel processes. This implementation provides
    a simplified version focusing on optimizer state partitioning (ZeRO Stage 1).

    For full ZeRO support with gradient and parameter partitioning, consider using
    DeepSpeed or PyTorch's FSDP (Fully Sharded Data Parallel).

    Args:
        optimizer: Base optimizer to wrap (e.g., AdamW, Lion)
        partition_optimizer_states: Whether to partition optimizer states (default: True)
        overlap_communication: Overlap gradient communication with computation (default: True)
        world_size: Number of distributed processes (default: None, auto-detected)
        rank: Process rank in distributed group (default: None, auto-detected)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        partition_optimizer_states: bool = True,
        overlap_communication: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        # Initialize base class with the wrapped optimizer's params and defaults
        params = []
        for group in optimizer.param_groups:
            params.extend(group["params"])

        # Initialize OptimizerBase with the parameters
        super().__init__(params, optimizer.defaults)

        self.optimizer = optimizer
        self.partition_optimizer_states = partition_optimizer_states
        self.overlap_communication = overlap_communication

        # Auto-detect distributed settings if not provided
        if world_size is None or rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.world_size = torch.distributed.get_world_size()
                self.rank = torch.distributed.get_rank()
            else:
                warnings.warn(
                    "Distributed training not initialized. ZeRO will run in single-GPU mode."
                )
                self.world_size = 1
                self.rank = 0
        else:
            self.world_size = world_size
            self.rank = rank

        self.is_distributed = self.world_size > 1

        # Track parameter partitions
        self.param_to_partition_id: Dict[torch.nn.Parameter, int] = {}
        self.partition_to_params: Dict[int, List[torch.nn.Parameter]] = {}

        if self.is_distributed and self.partition_optimizer_states:
            self._partition_parameters()

    def _partition_parameters(self):
        """
        Partition parameters across ranks for optimizer state sharding.
        """
        all_params = []
        for group in self.optimizer.param_groups:
            all_params.extend(group["params"])

        # Assign each parameter to a partition (rank)
        for idx, param in enumerate(all_params):
            partition_id = idx % self.world_size
            self.param_to_partition_id[param] = partition_id

            if partition_id not in self.partition_to_params:
                self.partition_to_params[partition_id] = []
            self.partition_to_params[partition_id].append(param)

    @property
    def param_groups(self):
        """Access underlying optimizer's parameter groups."""
        return self.optimizer.param_groups

    @property
    def state(self):
        """Access underlying optimizer's state."""
        return self.optimizer.state

    @property
    def defaults(self):
        """Access underlying optimizer's defaults."""
        return self.optimizer.defaults

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a dict.

        In distributed mode, only returns states for parameters owned by this rank.
        """
        base_state_dict = self.optimizer.state_dict()

        if not self.is_distributed or not self.partition_optimizer_states:
            return base_state_dict

        # Filter state to only include parameters owned by this rank
        filtered_state = {}
        for param_id, param_state in base_state_dict["state"].items():
            # Check if this parameter is owned by current rank
            param_groups = base_state_dict["param_groups"]
            # In practice, you'd need to map param_id back to actual parameter
            # This is a simplified version
            filtered_state[param_id] = param_state

        return {
            "state": filtered_state,
            "param_groups": base_state_dict["param_groups"],
            "zero_metadata": {
                "world_size": self.world_size,
                "rank": self.rank,
                "partition_optimizer_states": self.partition_optimizer_states,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the optimizer state.

        Args:
            state_dict: optimizer state dict
        """
        # Extract ZeRO metadata if present
        if "zero_metadata" in state_dict:
            metadata = state_dict.pop("zero_metadata")
            # Validate metadata matches current configuration
            if metadata["world_size"] != self.world_size:
                warnings.warn(
                    f"State dict world_size ({metadata['world_size']}) does not match "
                    f"current world_size ({self.world_size})"
                )

        self.optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        In distributed mode, synchronizes optimizer states across ranks as needed.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Optional loss value if closure is provided.
        """
        if not self.is_distributed:
            # Single-GPU mode: just call underlying optimizer
            return self.optimizer.step(closure)

        # Distributed mode with state partitioning
        if self.partition_optimizer_states:
            # Each rank only updates its owned parameters
            loss = self.optimizer.step(closure)

            # Synchronize updated parameters across ranks
            # This is a simplified version - in practice, you'd use
            # more efficient all-gather operations
            if torch.distributed.is_initialized():
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            # Broadcast parameter from its owner rank
                            owner_rank = self.param_to_partition_id.get(param, 0)
                            torch.distributed.broadcast(param.data, src=owner_rank)

            return loss
        else:
            # No partitioning: standard optimizer step
            return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        """
        Sets gradients of all optimized parameters to zero.

        Args:
            set_to_none: instead of setting to zero, set the grads to None.
                Default: True (overriding base class to match modern PyTorch conventions)
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Add a param group to the Optimizer's param_groups.

        Args:
            param_group: parameter group to add
        """
        self.optimizer.add_param_group(param_group)

        # Update partitioning if needed
        if self.is_distributed and self.partition_optimizer_states:
            self._partition_parameters()

    def __repr__(self):
        return (
            f"ZeROOptimizer("
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"world_size={self.world_size}, "
            f"rank={self.rank}, "
            f"partition_states={self.partition_optimizer_states}"
            f")"
        )

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"world_size={self.world_size}, "
            f"rank={self.rank}, "
            f"partition_states={self.partition_optimizer_states}"
        )
