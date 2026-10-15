# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: checkpoint save/load of actor +
#     critic + optimizer states + the resolved config (design #21, #23).
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

from marl.common.config import Config, config_from_dict, config_to_dict
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


def load_actor(path: str | Path) -> tuple[Config, Actor, Featurizer]:
    """Rebuild the trained actor (and its featurizer) from a checkpoint, for evaluation."""
    ckpt = load_checkpoint(path)
    cfg = config_from_dict(ckpt["config"])
    featurizer = Featurizer(cfg)
    actor, _ = build_networks(cfg, featurizer.obs_dim, featurizer.state_dim)
    actor.load_state_dict(ckpt["actor"])
    return cfg, actor, featurizer
