"""
Per-run infrastructure utilities.

This module standardizes:
- random seeds
- output directory layout
- config snapshots
- run metadata

Keeping these concerns here avoids each trainer/environment inventing its own
directory structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import random
import shutil

from config import AppConfig


@dataclass(frozen=True)
class RunContext:
    """
    Standard output locations shared with demo/RL code.

    "frozen=True" means the paths themselves should not be reassigned after the
    run starts.
    """
    run_dir: Path
    checkpoints_dir: Path
    metrics_path: Path
    config_snapshot_path: Path


def repo_root() -> Path:
    """
    Return repository root based on the current source layout.

    Current layout:
        repo/
          pybullet/
            runtime.py
    """
    return Path(__file__).resolve().parent.parent


def set_global_seed(seed: int) -> None:
    """
    Seed common RNG libraries when available.

    KeepAwayEnv.reset(seed=...) also seeds its own game randomness. Doing both
    makes the process-level experiment setup more reproducible.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Torch is optional until the RL implementation commits to it.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_run_context(
    cfg: AppConfig,
    source_config_path: str | Path,
) -> RunContext:
    """
    Create a unique output directory and preserve the exact input config.

    Example:
        outputs/
          20261014-124500-123456-marl-sports-simulator-demo/
            config.yaml
            run_metadata.json
            metrics.csv
            checkpoints/
    """
    # Microseconds reduce the chance of collision if runs start very quickly.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = f"{timestamp}-{cfg.project.name}-{cfg.run.mode}"

    output_root = Path(cfg.logging.output_dir)

    # Relative paths are interpreted from repository root rather than whatever
    # working directory happened to launch Python.
    if not output_root.is_absolute():
        output_root = repo_root() / output_root

    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"

    # exist_ok=False protects an old run from accidental overwrite.
    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Copy the source YAML so results can later be traced back to their exact
    # experiment configuration.
    snapshot = run_dir / "config.yaml"
    shutil.copy2(source_config_path, snapshot)

    metadata = {
        "project": cfg.project.name,
        "seed": cfg.project.seed,
        "mode": cfg.run.mode,
        "game": cfg.game.name,
        "algorithm": cfg.training.algorithm,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with (run_dir / "run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    return RunContext(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_path=run_dir / "metrics.csv",
        config_snapshot_path=snapshot,
    )
