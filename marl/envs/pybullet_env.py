# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-18
# Author:  Austin Beinder
# Usage:
#   - 2026-10-18: Generated this entire file: an adapter that makes the
#     PyBullet keep-away game (pybulletGame.KeepAwayEnv) speak docs/ICD.md
#     v0.2, so it can replace MockEnv in training and evaluation.
#   - 2026-10-18: Added the render_mode and print_collisions options.
# Prompt summary: Integrate the MARL layer with the PyBullet env through an
#   adapter with MockEnv's reset/step interface.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
"""ICD adapter around the PyBullet keep-away game (``pybulletGame.KeepAwayEnv``).

It has the same ``reset(seed, options)`` / ``step(actions)`` interface as ``MockEnv``
and translates in both directions:

* actions: ICD ``move`` in [-1, 1] (2-norm capped) x ``max_speed`` -> blue's velocity in m/s;
* state: the game's observation -> ICD ``s`` (``step``, ``blue``, ``red``, ``ball``);
* rewards are dropped: the RL layer computes its own (ICD §4);
* spawns come from ``sample_spawn``, like the mock, so both envs start from the same states;
* red possession ends the episode on the physics tick it happens (``red_win_time`` = one tick),
  and blue's "hold for 20 s" win is switched off: the ICD has no blue-win terminal (§7).

Limits (milestone 1): keepaway only, 1 blue vs 1 red, no kicking. Possession is contact-based
in the sim (player edge touching ball edge), not ``possession_radius``.

All instances share pybullet's single default connection (the entity classes use it), so only
one instance can be mid-episode at a time: ``reset`` claims the simulation, and stepping an
instance whose simulation another instance has since reset raises instead of silently
continuing from the wrong state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marl.common.config import SharedConfig
from marl.common.spawn import RESET_OPTION_KEYS, sample_spawn
from marl.envs.mock_env import parse_actions
from pybulletGame import Entity, KeepAwayConfig, KeepAwayEnv

_owner: "PyBulletEnv | None" = None  # the instance whose episode the shared simulation holds


class PyBulletEnv:
    """ICD-conformant wrapper around ``KeepAwayEnv``. ``reset`` and ``step`` speak plain Python types."""

    def __init__(self, cfg: SharedConfig, render_mode: str | None = None, print_collisions: bool = False):
        """render_mode="human" opens the pybullet GUI (only if this process has no pybullet connection
        yet). print_collisions turns on the sim's collision debug prints, for the whole process."""
        if cfg.scenario != "keepaway":
            raise NotImplementedError(f"the PyBullet env only implements keepaway, not {cfg.scenario!r}")
        if (cfg.n_blue, cfg.n_red) != (1, 1):
            raise NotImplementedError(f"the PyBullet env is 1v1 only, got {cfg.n_blue}v{cfg.n_red}")
        self.cfg = cfg
        self.n_blue, self.n_red = cfg.n_blue, cfg.n_red
        Entity.print_collisions = print_collisions
        episode_time = cfg.max_episode_steps * cfg.control_k * cfg.physics_dt
        self._game = KeepAwayEnv(
            KeepAwayConfig(
                blue_max_speed=cfg.max_speed,
                red_max_speed=cfg.red_max_speed,
                possession_speed_multiplier=cfg.carry_speed_factor,
                player_radius=cfg.player_radius,
                red_win_time=cfg.physics_dt,  # one tick: red possession is terminal at once (ICD §3, §7)
                blue_win_time=episode_time + 1.0,  # never reached: the ICD has no blue win
                max_game_time=episode_time,
                field_length=cfg.field_size[0],
                field_width=cfg.field_size[1],
                action_repeat=cfg.control_k,
            ),
            render_mode=render_mode,
        )
        if not np.isclose(self._game.time_step, cfg.physics_dt):
            raise ValueError(f"physics_dt {cfg.physics_dt} doesn't match pybullet's time step {self._game.time_step}")
        self._ready = False

    # --- API (ICD §4) ----------------------------------------------------------
    def reset(self, seed: int, options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        global _owner
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        opts = self.cfg.start.to_reset_options()
        if options:
            unknown = set(options) - RESET_OPTION_KEYS
            if unknown:
                raise ValueError(f"unknown reset option(s) {sorted(unknown)}")
            opts.update(options)
        sp = sample_spawn(
            np.random.default_rng(seed),
            scenario=self.cfg.scenario,
            n_blue=self.n_blue,
            n_red=self.n_red,
            field_size=self.cfg.field_size,
            player_radius=self.cfg.player_radius,
            possession_radius=self.cfg.possession_radius,
            options=opts,
        )
        self._game.reset(
            seed=seed,  # seeds the steal roll in Ball, so episodes replay exactly
            options={
                "blue_position": sp.blue_pos[0],
                "red_position": sp.red_pos[0],
                "ball_position": sp.ball_pos,
                "blue_has_ball": sp.holder is not None,
            },
        )
        _owner = self
        self._step = 0
        self._ready = True
        return self._state(), self._info([])

    def step(self, actions: dict[str, Any]) -> tuple[dict, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("call reset() before step(); there is no auto-reset (ICD §4)")
        if _owner is not self:
            raise RuntimeError(
                "another PyBulletEnv reset the shared pybullet simulation since this one's reset(); "
                "only one PyBulletEnv can be mid-episode at a time"
            )
        move, triggers = parse_actions(actions, self.n_blue)
        if triggers[0] == 1 and self._game.ball.possession_player is self._game.blue:
            raise NotImplementedError(
                "the PyBullet adapter doesn't implement kicking yet (deferred with kick_enabled, #21); "
                "the RL side sends kick_trigger = 0 at milestone 1"
            )

        _, _, terminated, _, _ = self._game.step(move[0] * self.cfg.max_speed)
        self._step += 1
        truncated = (not terminated) and self._step >= self.cfg.max_episode_steps
        if terminated or truncated:
            self._ready = False  # no auto-reset; the returned state is the true final state
        events = [
            {"type": "possession_change", "from": self._holder_id(old), "to": self._holder_id(new)}
            for _, old, new in self._game.possession_events
        ]
        return self._state(), bool(terminated), truncated, self._info(events)

    def close(self) -> None:
        """Disconnects pybullet, which ends every PyBulletEnv in this process."""
        self._ready = False
        self._game.close()

    # --- ICD plumbing -------------------------------------------------------------
    def _holder_id(self, player: Any) -> tuple[str, int] | None:
        if player is None:
            return None
        return ("blue", 0) if player is self._game.blue else ("red", 0)

    def _state(self) -> dict:
        game = self._game
        holder = game.ball.possession_player

        def xy(v: np.ndarray) -> list[float]:
            return [float(v[0]), float(v[1])]

        def entity(player: Any) -> dict:
            return {"id": 0, "pos": xy(player.position), "vel": xy(player.velocity), "has_possession": holder is player}

        return {
            "step": self._step,
            "blue": [entity(game.blue)],
            "red": [entity(game.red)],
            "ball": {"pos": xy(game.ball.position), "vel": xy(game.ball.velocity)},
        }

    def _info(self, events: list[dict]) -> dict:
        return {"events": events, "kick_applied": [False] * self.n_blue}
### END AI GENERATED
