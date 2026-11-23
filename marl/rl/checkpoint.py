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
#   - 2026-11-07: warm_start, which also works across team sizes (e.g. 1v1 ->
#     2v1) by growing the first layers' input columns. (Austin Beinder)
#   - 2026-11-07: warm_start keeps the critic only when the task and reward are
#     the same (e.g. not goto_target -> keep-away). (Austin Beinder)
#   - 2026-11-09: Teams may also shrink (e.g. 1v1 -> 3v0 for pass practice):
#     the removed players' input columns are dropped. (Austin Beinder)
#   - 2026-11-12: rl.new_player_init = "random" starts new players' columns
#     at the fresh network's init instead of zero. (Austin Beinder)
#   - 2026-11-13: warm_start also adds (or drops) per-slot features, e.g. the
#     rl.passer_obs flags, like new players' columns. (Austin Beinder)
#   - 2026-11-13: ... and the rl.call_enabled flags; a new call head keeps its
#     fresh init. (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters; no
#     behaviour change. (Austin Beinder)
#   - 2026-11-23: warm_start maps the soccer goals block too, so a keep-away
#     actor warm-starts into soccer. (Austin Beinder)
# ==============================================================================
"""Checkpoints (#23): actor + critic weights, optimizer states, progress counters
and the resolved config, so a checkpoint is self-describing (``load_actor`` can
rebuild the right network sizes from it). Everything is a tensor or a plain
Python value, so loading works with ``torch.load(weights_only=True)``.

Warm start (#21): a later actor with extra heads loads ``ckpt["actor"]`` with
``load_state_dict(strict=False)``; only the new heads stay freshly initialized.

Warm start across team sizes (``warm_start``, 2026-11-07): the actor's and the critic's inputs
are built from per-player blocks (``obs_blocks``, ``state_blocks``). With more players, every
old input column moves to the same slot in the new layout, and the new slots' columns get zero
weight. So at first the network gives exactly the old outputs and ignores the new players;
those columns still get gradients, so it learns to use them.

That learning can be slow (2026-11-12): a 1v0 go_to_target actor grown to 3v0 still had teammate
columns ~6x weaker than the ball's after 180k steps, and acted as if alone. With
``rl.new_player_init = "random"`` the new columns instead keep the fresh network's own random
init, so the outputs change a little at first but the new players are visible from the start.

Teams may also shrink (2026-11-09): a player slot the new layout doesn't have loses its columns,
so the network forgets those weights (e.g. red's, going 1v1 -> 3v0; they come back as zero
columns if red returns later). Every slot both layouts have keeps its weights.

Slots may also gain features at their end (2026-11-13: ``rl.passer_obs`` adds a flag to the self
block, each teammate slot and each blue in the critic's state). The new features' columns start
like new players' (zero, or random with ``rl.new_player_init = "random"``); dropped ones go.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np  # AI GENERATED
import torch

from marl.common.config import Config, config_to_dict, saved_config_from_dict
from marl.common.config import Config, config_from_dict, config_to_dict

from marl.rl.networks import Actor, Critic, build_networks
from marl.rl.obs import goal_obs_for, obs_blocks  # AI GENERATED (goal_obs_for 2026-11-23)
from marl.rl.state import state_blocks  # AI GENERATED
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


### AI GENERATED
def warm_start(
    actor: Actor, critic: Critic, ckpt: dict[str, Any], cfg: Config
) -> tuple[list[str], bool]:
    """Load a checkpoint's weights into networks built for ``cfg``. Returns the actor parameters
    the checkpoint didn't have (they keep their fresh init, e.g. new kick heads), and whether the
    critic was loaded.

    The critic is loaded only if the scenario and the reward are the same: it predicts returns of
    one reward, so after goto_target (-distance) it would be wrong for keep-away (+1 per step),
    and it starts fresh instead. The actor always loads.

    If the team sizes (or, 2026-11-13, ``rl.passer_obs``) differ from the checkpoint's config, the
    first layers' input columns are remapped: new players' (and features') columns are zero (or,
    with ``cfg.rl.new_player_init = "random"``, keep the networks' fresh init, 2026-11-12),
    removed players' columns are dropped (see the module docstring). Raises if the critic would
    load but its input mode changed.
    """
    old = config_from_dict(ckpt["config"])
    actor_sd, critic_sd = dict(ckpt["actor"]), dict(ckpt["critic"])
    o, n = old.shared, cfg.shared
    load_critic = old.shared.scenario == cfg.shared.scenario and old.rl.reward == cfg.rl.reward
    if load_critic and old.rl.critic_input != cfg.rl.critic_input:
        raise ValueError(
            f"can't warm start a {cfg.rl.critic_input!r} critic from a {old.rl.critic_input!r} one"
        )
    ### AI GENERATED (2026-11-13: the layouts may also differ in the passer / call flags)
    # AI GENERATED (2026-11-23: and the soccer goals block)
    old_obs = obs_blocks(o.n_blue, o.n_red, old.rl.passer_obs, old.rl.call_enabled, goal_obs_for(o))
    new_obs = obs_blocks(n.n_blue, n.n_red, cfg.rl.passer_obs, cfg.rl.call_enabled, goal_obs_for(n))
    old_state = state_blocks(o.n_blue, o.n_red, old.rl.passer_obs, old.rl.call_enabled)
    new_state = state_blocks(n.n_blue, n.n_red, cfg.rl.passer_obs, cfg.rl.call_enabled)
    if (old_obs, old_state) != (new_obs, new_state):
        obs_cols = _column_map(old_obs, new_obs)
        state_cols = _column_map(old_state, new_state)
    ### END AI GENERATED
        random = cfg.rl.new_player_init == "random"  # AI GENERATED (2026-11-12)
        actor_sd["trunk.0.weight"] = _grow(
            actor_sd["trunk.0.weight"], *obs_cols,
            fresh=actor.state_dict()["trunk.0.weight"] if random else None)
        if load_critic:
            critic_cols = state_cols if cfg.rl.critic_input == "global" else obs_cols
            critic_sd["body.0.weight"] = _grow(
                critic_sd["body.0.weight"], *critic_cols,
                fresh=critic.state_dict()["body.0.weight"] if random else None)

    loaded = actor.load_state_dict(actor_sd, strict=False)  # missing heads keep their fresh init
    if loaded.unexpected_keys:
        raise ValueError(
            f"the checkpoint's actor has parameters this config doesn't: {loaded.unexpected_keys}"
        )
    if load_critic:
        critic.load_state_dict(critic_sd)
    return list(loaded.missing_keys), load_critic


def _column_map(old_blocks, new_blocks) -> tuple[np.ndarray, np.ndarray, int]:
    """The old input columns that survive, where each goes in the new layout (slot k of a block
    stays slot k; slots past the new count are dropped, 2026-11-09; feature f of a slot stays
    feature f, and features past the new width are dropped, 2026-11-13), and the new input size."""
    kept, mapping, old_start, new_start = [], [], 0, 0
    for (name, width, old_slots), (new_name, new_width, new_slots) in zip(old_blocks, new_blocks,
                                                                          strict=True):
        assert name == new_name
        ### AI GENERATED (2026-11-13: per-slot features may be added or dropped at the end)
        for slot in range(min(old_slots, new_slots)):
            for f in range(min(width, new_width)):
                kept.append(old_start + slot * width + f)
                mapping.append(new_start + slot * new_width + f)
        ### END AI GENERATED
        old_start += width * old_slots
        new_start += new_width * new_slots
    return np.asarray(kept, dtype=np.int64), np.asarray(mapping, dtype=np.int64), new_start


def _grow(weight: torch.Tensor, kept: np.ndarray, columns: np.ndarray, new_inputs: int,
          fresh: torch.Tensor | None = None) -> torch.Tensor:
    """A first-layer weight [out, old_in] -> [out, new_in]: kept columns moved, new ones zero, or
    (2026-11-12) taken from ``fresh``, the new network's own freshly initialized weight."""
    if fresh is not None:
        assert fresh.shape == (weight.shape[0], new_inputs)
    grown = (torch.zeros(weight.shape[0], new_inputs, dtype=weight.dtype) if fresh is None
             else fresh.detach().clone().to(weight.dtype))
    grown[:, torch.as_tensor(columns)] = weight[:, torch.as_tensor(kept)]
    return grown
### END AI GENERATED


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
