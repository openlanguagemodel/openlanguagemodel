"""
The on-disk format of OLM training checkpoints.

A training checkpoint is one ``torch.save`` file holding everything needed to
continue a run exactly where it stopped: model and optimizer state, scheduler
and AMP scaler state, the Trainer's step counters and data position, every
random number generator, and callback state. ``Trainer.save_checkpoint`` and
``Trainer.load_checkpoint`` produce and consume it; this module holds the
pieces that do not need a Trainer: reading (including the older formats),
atomic writing, and finding the newest checkpoint in a directory.

Format version 1 (the keys of the saved dict):

    format_version   1
    olm_version      OLM version that wrote the file
    torch_version    torch version that wrote the file
    model            model state dict without "module." / "_orig_mod." prefixes
    optimizer        optimizer state dict, or None when it was not saved
    scheduler        scheduler state dict, or None when there is no scheduler
    scheduler_class  class name of the scheduler, or None
    scaler           GradScaler state dict ({} when AMP scaling is off)
    trainer          step counters, token count, schedule length and the data
                     position (epoch, optimizer steps and batches consumed in it)
    rng              one entry per rank, each holding the random number
                     generator states at save time ("current") and at the
                     start of the current epoch ("epoch_start")
    callbacks        state of each callback, keyed by class name
    losses           the full training-loss history

Files written by version 1 also carry the pre-version-1 key names
(``model_state_dict``, ``optimizer_state_dict``, ``scheduler_state_dict``,
``scaler_state_dict``, ``step``, ``global_step``, ``epoch``) as aliases of the
same objects, so scripts written for the old ``CheckpointCallback`` format keep
working. The aliases cost no disk space and will be removed in a future release.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1

# Prefixes that wrappers add to every state-dict key: DistributedDataParallel
# and DataParallel add "module.", torch.compile adds "_orig_mod.".
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")

_STEP_CHECKPOINT_PATTERN = re.compile(r"^step_(\d+)\.pt$")


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the user's model from inside DDP, DataParallel and ``torch.compile`` wrappers.

    State dicts taken from the unwrapped model have no ``module.`` or
    ``_orig_mod.`` prefixes, so they load into a plain model. FSDP is not
    unwrapped: its parameters are sharded and need FSDP's own state-dict APIs.

    Args:
        model: A model, possibly wrapped (repeatedly).

    Returns:
        The innermost model.
    """
    while True:
        if isinstance(
            model,
            (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel),
        ):
            model = model.module
        elif hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
            model = model._orig_mod
        else:
            return model


def _strip_wrapper_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip "module." / "_orig_mod." when every key carries the prefix."""
    stripped = True
    while stripped and state_dict:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if all(key.startswith(prefix) for key in state_dict):
                state_dict = {key[len(prefix) :]: v for key, v in state_dict.items()}
                stripped = True
    return state_dict


def _normalize_legacy_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a checkpoint written before format version 1 to the version 1 layout.

    Two older layouts exist. ``CheckpointCallback`` (and ``WandBCallback``
    artifacts) used ``model_state_dict`` / ``optimizer_state_dict`` /
    ``scheduler_state_dict`` / ``scaler_state_dict`` / ``step`` / ``epoch``.
    ``FSDPTrainer.save_checkpoint`` used ``model`` / ``optimizer`` /
    ``scheduler`` / ``global_step`` / ``epoch``, and its optimizer entry held
    only rank 0's shard, so it is dropped here. Neither layout recorded a data
    position or RNG state.
    """
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        global_step = checkpoint.get("step", 0)
    elif "model" in checkpoint and "global_step" in checkpoint:
        model_state = checkpoint["model"]
        optimizer_state = None
        if checkpoint.get("optimizer") is not None:
            warnings.warn(
                "This checkpoint uses the FSDPTrainer layout from before format "
                "version 1. Its optimizer entry holds only rank 0's shard, so "
                "optimizer state is not restored; the optimizer starts fresh.",
                stacklevel=3,
            )
        scheduler_state = checkpoint.get("scheduler")
        global_step = checkpoint["global_step"]
    else:
        raise ValueError(
            "Unrecognized checkpoint: expected an OLM training checkpoint with a "
            "'format_version' key, or the older 'model_state_dict' / 'global_step' "
            f"layouts. Found keys: {sorted(checkpoint)[:10]}"
        )

    return {
        "format_version": 0,
        "model": _strip_wrapper_prefixes(model_state),
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "scheduler_class": None,
        "scaler": checkpoint.get("scaler_state_dict") or {},
        "trainer": {
            "global_step": global_step,
            "epoch": checkpoint.get("epoch", 0),
        },
        "rng": None,
        "callbacks": {},
        "losses": list(checkpoint.get("losses") or []),
    }


def read_checkpoint(
    path: Union[str, os.PathLike],
    map_location: Union[str, torch.device] = "cpu",
    weights_only: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint from disk in the format version 1 layout.

    Older ``CheckpointCallback`` and ``FSDPTrainer`` files are converted, with
    ``module.`` / ``_orig_mod.`` prefixes removed from the model keys. To load
    just the weights into a model::

        model.load_state_dict(read_checkpoint("checkpoints/best_model.pt")["model"])

    Args:
        path: Checkpoint file.
        map_location: Where to put tensors while loading. Defaults to ``"cpu"``;
            ``load_state_dict`` then copies them to the model's device.
        weights_only: Passed to ``torch.load``. ``True`` (default) restricts
            unpickling to tensors, primitive types and containers. ``False``
            can execute arbitrary code stored in the file, so use it only for
            files you trust, for example when a custom scheduler stored Python
            objects in its state.

    Returns:
        The checkpoint dict (see the module docstring for its keys). Converted
        older files have ``format_version`` 0 and no data position or RNG state.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"{path} does not contain a training checkpoint (found "
            f"{type(checkpoint).__name__}). Models written by Block.save() load "
            "with olm.nn.structure.load_model()."
        )

    version = checkpoint.get("format_version")
    if version is None:
        return _normalize_legacy_checkpoint(checkpoint)
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} uses checkpoint format version {version}, but this OLM "
            f"release reads versions up to {CHECKPOINT_FORMAT_VERSION}. Upgrade "
            "OLM to load it."
        )
    return checkpoint


def add_legacy_aliases(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add the pre-version-1 key names as aliases of the version 1 entries.

    The aliases point at the same objects, and ``torch.save`` stores an object
    referenced twice only once, so they do not grow the file.
    """
    trainer_state = checkpoint["trainer"]
    checkpoint["model_state_dict"] = checkpoint["model"]
    if checkpoint["optimizer"] is not None:
        checkpoint["optimizer_state_dict"] = checkpoint["optimizer"]
    if checkpoint["scheduler"] is not None:
        checkpoint["scheduler_state_dict"] = checkpoint["scheduler"]
    checkpoint["scaler_state_dict"] = checkpoint["scaler"]
    checkpoint["step"] = trainer_state["global_step"]
    checkpoint["global_step"] = trainer_state["global_step"]
    checkpoint["epoch"] = trainer_state["epoch"]
    return checkpoint


def write_checkpoint(checkpoint: Dict[str, Any], path: Union[str, os.PathLike]) -> Path:
    """
    Write ``checkpoint`` to ``path`` atomically.

    The file is first written next to ``path`` under a temporary name and then
    renamed over it, so an interrupted save never leaves a truncated file where
    a good checkpoint used to be.

    Args:
        checkpoint: The dict to save.
        path: Destination file. Parent directories are created.

    Returns:
        ``path`` as a ``Path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    return path


def find_latest_checkpoint(
    checkpoint_dir: Union[str, os.PathLike],
) -> Optional[Path]:
    """
    Return the ``step_<N>.pt`` file with the largest ``N`` in ``checkpoint_dir``.

    Args:
        checkpoint_dir: Directory written by ``CheckpointCallback``.

    Returns:
        The newest step checkpoint, or ``None`` when the directory is missing
        or holds none. ``best_model.pt`` is never returned.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    latest_step, latest_path = -1, None
    for path in checkpoint_dir.iterdir():
        match = _STEP_CHECKPOINT_PATTERN.match(path.name)
        if match and int(match.group(1)) > latest_step:
            latest_step, latest_path = int(match.group(1)), path
    return latest_path
