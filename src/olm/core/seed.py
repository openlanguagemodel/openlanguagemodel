"""
Seeding and random-state helpers for reproducible training.

Every source of randomness in a training run draws from one of four random
number generators: Python's ``random`` module, NumPy's global generator, the
torch CPU generator, and one torch generator per CUDA device. These helpers
seed all of them together, capture and restore their state for checkpoints,
and seed ``DataLoader`` worker processes and shuffling.
"""

import os
import random
import warnings
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(
    seed: int, deterministic: bool = False, warn_only: bool = True
) -> int:
    """
    Seed Python, NumPy and torch on the CPU and every CUDA device.

    Call this before you build the model: weight initialization draws from the
    torch generator, so seeding afterwards makes dropout and shuffling
    reproducible but not the initial weights.

    Args:
        seed: The seed shared by every generator.
        deterministic: Also ask PyTorch for deterministic kernels. This sets
            ``torch.use_deterministic_algorithms(True, warn_only=warn_only)``,
            ``torch.backends.cudnn.deterministic = True``,
            ``torch.backends.cudnn.benchmark = False`` and, if it is unset, the
            ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` environment variable that cuBLAS
            needs for deterministic matrix multiplies. Deterministic kernels
            can be slower. When ``False`` (default) these settings are left
            untouched.
        warn_only: Passed to ``torch.use_deterministic_algorithms``. With
            ``True`` (default), an operation that has no deterministic
            implementation warns instead of raising.

    Returns:
        The seed, for logging.

    Example:
        >>> from olm.core import seed_everything
        >>> seed_everything(42)  # before the model is built
        42
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    # torch.manual_seed also seeds every CUDA device.
    torch.manual_seed(seed)

    if deterministic:
        # cuBLAS reads this when it creates its first handle, so it only takes
        # effect if no CUDA matrix multiply has run yet in this process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def seed_worker(worker_id: int) -> None:
    """
    Seed Python and NumPy inside a ``DataLoader`` worker process.

    PyTorch already gives each worker its own torch seed (``base_seed +
    worker_id``), but it does not reseed Python's ``random`` module or NumPy,
    so without this every worker would repeat the same random numbers. Pass it
    as ``worker_init_fn``.

    Args:
        worker_id: The worker index. Unused; the torch seed already encodes it.

    Example:
        >>> from olm.core import make_generator, seed_worker
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=8,
        ...     num_workers=2,
        ...     worker_init_fn=seed_worker,
        ...     generator=make_generator(42),
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """
    Create a CPU ``torch.Generator`` seeded with ``seed``.

    Pass it as ``DataLoader(generator=...)`` to give shuffling and worker
    seeds their own random stream. The shuffle order then no longer depends
    on how many random numbers the rest of the program drew before the epoch
    started.

    Args:
        seed: Seed for the generator.

    Returns:
        A seeded ``torch.Generator`` on the CPU.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_rng_state() -> Dict[str, Any]:
    """
    Capture the state of every random number generator.

    The result only contains tensors and plain Python values, so it can be
    stored in a checkpoint that loads with ``torch.load(weights_only=True)``.
    CUDA states are captured only when CUDA has been initialized in this
    process, so a CPU run never initializes CUDA as a side effect.

    Returns:
        A dict with the keys ``"python"``, ``"numpy"``, ``"torch"`` and, once
        CUDA is in use, ``"cuda"`` (one state per device).
    """
    kind, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    state = {
        "python": random.getstate(),
        # NumPy's key array becomes a list so the state loads under
        # torch.load(weights_only=True).
        "numpy": (kind, keys.tolist(), pos, has_gauss, cached_gaussian),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore generator states captured by :func:`get_rng_state`.

    Keys missing from ``state`` are left alone. If the checkpoint was written
    on a machine with a different number of CUDA devices, only the devices
    present on both are restored and a warning is emitted.

    Args:
        state: A dict returned by :func:`get_rng_state`.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        kind, keys, pos, has_gauss, cached_gaussian = state["numpy"]
        np.random.set_state(
            (kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian)
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        device_count = torch.cuda.device_count()
        if len(cuda_states) != device_count:
            warnings.warn(
                f"The saved RNG state has {len(cuda_states)} CUDA device(s) but "
                f"{device_count} are visible; restoring the first "
                f"{min(len(cuda_states), device_count)}.",
                stacklevel=2,
            )
        for device, cuda_state in enumerate(cuda_states[:device_count]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device)
