"""
Generic neural-network checkpoint helpers.

TEAM NOTES
----------
This file does NOT define a neural network and does NOT assume PPO/IPPO/MAPPO.
It only provides optional PyTorch save/load helpers that a future trainer can
use.

Torch is imported lazily because it is not currently listed in the repository
dependencies. If the RL issue chooses a different framework, this helper can be
replaced without changing main.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_torch_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    *,
    episode: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save model weights plus optional optimizer/training state.

    Saving optimizer state is useful when resuming training because optimizers
    such as Adam maintain internal momentum/statistics beyond model weights.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for checkpoint I/O. "
            "Add torch after the RL implementation standardizes on PyTorch."
        ) from exc

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if episode is not None:
        payload["episode"] = episode

    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)
    return path


def load_torch_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load model weights and optional optimizer state from a checkpoint."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for checkpoint I/O."
        ) from exc

    checkpoint = torch.load(
        Path(path),
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint
