# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Created this package marker.
#   - 2026-10-18: Added make_env (the run.env selector) and SHARED_SIM_ENVS,
#     with render / print_collisions options.
#     (Austin Beinder)
#   - 2026-10-31: Removed SHARED_SIM_ENVS: every PyBulletEnv now has its own
#     simulation. (Austin Beinder)
# Prompt summary: Implement Phase 3 of the issue #7 plan (RL layer + mock env).
# Human review:   Pending review
# ==============================================================================
"""Environments: the ICD mock env and the PyBullet adapter, picked by ``run.env``."""

### AI GENERATED
from __future__ import annotations

from typing import Any

from marl.common.config import Config


def make_env(cfg: Config, *, render: bool = False, print_collisions: bool = False) -> Any:
    """A new env for ``cfg.run.env``, built from ``cfg.shared``.

    render opens the PyBullet GUI (one per process); print_collisions turns on the sim's collision
    debug prints, for every PyBullet env in the process. Both are PyBullet-only (the mock has no GUI
    and no collisions).
    """
    if cfg.run.env == "mock":
        if render:
            raise ValueError("the mock env has no GUI; render needs run.env = 'pybullet'")
        from marl.envs.mock_env import MockEnv

        return MockEnv(cfg.shared)
    if cfg.run.env == "pybullet":
        from marl.envs.pybullet_env import PyBulletEnv  # imported lazily: pulls in pybullet

        return PyBulletEnv(cfg.shared, render_mode="human" if render else None, print_collisions=print_collisions)
    raise ValueError(f"unknown run.env {cfg.run.env!r}")
### END AI GENERATED
