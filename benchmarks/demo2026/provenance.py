"""Environment and provenance capture for reportable benchmark runs.

Every result record produced by this package embeds the dictionary returned
by :func:`capture_environment` so that raw results are self-describing and can
be audited long after the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch


def _run_git(args: list[str], cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = _run_git(["rev-parse", "--show-toplevel"], cwd=here)
    return root or os.path.abspath(os.path.join(here, "..", ".."))


def git_commit() -> Optional[str]:
    return _run_git(["rev-parse", "HEAD"], cwd=repo_root())


def git_is_dirty() -> bool:
    status = _run_git(["status", "--porcelain"], cwd=repo_root())
    if status is None:
        return True
    return bool(status.strip())


def _optional_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def capture_environment() -> Dict[str, Any]:
    """Capture interpreter, library, CUDA, and git state for a run record."""
    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": _optional_version("transformers"),
        "numpy_version": _optional_version("numpy"),
        "olm_version": _optional_version("olm"),
        "litgpt_version": _optional_version("litgpt"),
        "olm_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
    }

    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_models"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        try:
            env["nccl_version"] = ".".join(str(v) for v in torch.cuda.nccl.version())
        except Exception:
            env["nccl_version"] = None
        try:
            env["driver_version"] = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()[0]
            )
        except Exception:
            env["driver_version"] = None
    else:
        env["gpu_count"] = 0
        env["gpu_models"] = []
        env["nccl_version"] = None
        env["driver_version"] = None

    relevant_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "OMP_NUM_THREADS",
        "TORCH_COMPILE_DISABLE",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]
    env["env_vars"] = {k: os.environ.get(k) for k in relevant_env_vars if k in os.environ}
    return env


def config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of a config dictionary for cross-run identity checks."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def require_clean_worktree(allow_dirty: bool = False) -> None:
    """Refuse reportable runs from an uncommitted state unless overridden."""
    if git_is_dirty() and not allow_dirty:
        raise RuntimeError(
            "Refusing reportable run: git worktree is dirty. Commit your changes "
            "first, or pass --allow-dirty for a non-reportable debug run."
        )


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
