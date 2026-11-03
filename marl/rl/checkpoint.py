# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: checkpoint save/load of actor +
#     critic + optimizer states + the resolved config (design #21, #23).
#   - 2026-11-01: load_actor reads the saved config with saved_config_from_dict,
#     so checkpoints from before run.env existed rebuild on the mock (task 3).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
"""Checkpoints (#23): actor + critic weights, optimizer states, progress counters
and the resolved config, so a checkpoint is self-describing (``load_actor`` can
rebuild the right network sizes from it). Everything is a tensor or a plain
Python value, so loading works with ``torch.load(weights_only=True)``.

Warm start (#21): a later actor with extra heads loads ``ckpt["actor"]`` with
``load_state_dict(strict=False)``; only the new heads stay freshly initialized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from marl.common.config import Config, config_to_dict, saved_config_from_dict
from marl.rl.networks import Actor, Critic, build_networks
from marl.rl.ppo import PPOUpdater
from marl.rl.rollout import Featurizer

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    actor: Actor,
    critic: Critic,
    updater: PPOUpdater,
    iteration: int,
    env_steps: int,
    cfg: Config,
) -> None:
    torch.save(
        {
            "format": FORMAT_VERSION,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": updater.actor_opt.state_dict(),
            "critic_opt": updater.critic_opt.state_dict(),
            "iteration": iteration,
            "env_steps": env_steps,
            "config": config_to_dict(cfg),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(path, weights_only=True)
    if ckpt.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint format {ckpt.get('format')!r}")
    return ckpt


# AI CITATION: Claude Code (Claude Opus 5.5) | 2026-10-11 | JC Stinson III (LifeHackerIII)
# Usage: Generated with this file (see the file-level citation).
# Modified: 2026-11-01 - The config loads via saved_config_from_dict, so a
#   checkpoint with no run.env rebuilds on the mock (backlog task 3).
def load_actor(path: str | Path) -> tuple[Config, Actor, Featurizer]:
    """Rebuild the trained actor (and its featurizer) from a checkpoint, for evaluation."""
    ckpt = load_checkpoint(path)
    cfg = saved_config_from_dict(ckpt["config"])  # AI GENERATED (was config_from_dict)
    featurizer = Featurizer(cfg)
    actor, _ = build_networks(cfg, featurizer.obs_dim, featurizer.state_dim)
    actor.load_state_dict(ckpt["actor"])
    return cfg, actor, featurizer
