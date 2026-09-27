"""Parity-suite fixtures.

The suite must be importable as ``tests.parity`` from the repository root, and
must run identically whether invoked as ``pytest tests/parity`` or as part of a
full ``pytest`` run.
"""

from __future__ import annotations

import contextlib

import pytest
import torch


@contextlib.contextmanager
def preserved_torch_globals():
    """Snapshot the process-global torch settings this suite mutates, and restore them.

    ``run_parity`` calls ``set_determinism()``, which enables deterministic
    algorithms, disables cuDNN autotuning and pins the thread count. All three
    are *process-global*: left set, they change kernel selection, warning
    behaviour and GPU performance for every test that happens to run after this
    package in a full ``pytest`` session. Restoring only the thread count fixes
    the cheapest of the three and leaks the other two.

    Deliberately restores the recorded values rather than library defaults, so
    running under a caller that itself wanted determinism on stays correct.
    """
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        torch.set_num_threads(previous_threads)


@pytest.fixture(autouse=True)
def _deterministic_cpu():
    """Pin threading and algorithm choice around every parity test."""
    with preserved_torch_globals():
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        yield
