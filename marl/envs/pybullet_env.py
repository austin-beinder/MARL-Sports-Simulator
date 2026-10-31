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
#   - 2026-10-31: N blue vs M red, the goto_target scenario, teammate handoff
#     off (ICD §8), and the set_state_for_testing hook (same as MockEnv's).
#   - 2026-10-31: Each env now has its own simulation, so the one-env-mid-
#     episode guard is gone.
# Prompt summary: Integrate the MARL layer with the PyBullet env through an
#   adapter with MockEnv's reset/step interface.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
"""ICD adapter around the PyBullet keep-away game (``pybulletGame.KeepAwayEnv``).

It has the same ``reset(seed, options)`` / ``step(actions)`` interface as ``MockEnv``
and translates in both directions:

* actions: ICD ``move`` in [-1, 1] (2-norm capped) x ``max_speed`` -> each blue's velocity in m/s;
* state: the game's players and ball -> ICD ``s`` (``step``, ``blue``, ``red``, ``ball``);
* rewards are dropped: the RL layer computes its own (ICD §4);
* spawns come from ``sample_spawn``, like the mock, so both envs start from the same states;
* red possession ends the episode on the physics tick it happens (``red_win_time`` = one tick),
  and blue's "hold for 20 s" win is switched off: the ICD has no blue-win terminal (§7);
* only an opponent can take the ball from its carrier (teammate handoff off, ICD §8);
* goto_target: no red, and the ball is an immobile target nobody can touch or hold (§9).

Limits: no kicking yet. Possession is contact-based in the sim (player edge touching ball edge,
or an opponent touching the carrier), not ``possession_radius``.

Each instance has its own pybullet simulation (``KeepAwayEnv`` opens a connection per game), so
any number can be mid-episode in one process. Only one can render (pybullet allows one GUI).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marl.common.config import SharedConfig
from marl.common.spawn import RESET_OPTION_KEYS, sample_spawn
from marl.envs.mock_env import parse_actions
from pybulletGame import Entity, KeepAwayConfig, KeepAwayEnv

_UNSET = object()


class PyBulletEnv:
    """ICD-conformant wrapper around ``KeepAwayEnv``. ``reset`` and ``step`` speak plain Python types."""

    def __init__(self, cfg: SharedConfig, render_mode: str | None = None, print_collisions: bool = False):
        """render_mode="human" opens the pybullet GUI (only if this process has no pybullet connection
        yet). print_collisions turns on the sim's collision debug prints, for the whole process."""
        if cfg.scenario not in ("keepaway", "goto_target"):
            raise NotImplementedError(f"the PyBullet env doesn't implement scenario {cfg.scenario!r}")
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
                n_blue=cfg.n_blue,
                n_red=cfg.n_red,
                teammate_handoff=False,  # only a kick releases the ball (ICD §8)
                ball_is_target=cfg.scenario == "goto_target",  # immobile, untouchable target (ICD §9)
            ),
            render_mode=render_mode,
        )
        if not np.isclose(self._game.time_step, cfg.physics_dt):
            raise ValueError(f"physics_dt {cfg.physics_dt} doesn't match pybullet's time step {self._game.time_step}")
        self._ready = False

    # --- API (ICD §4) ----------------------------------------------------------
    def reset(self, seed: int, options: dict[str, Any] | None = None) -> tuple[dict, dict]:
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
        self._seed = seed  # seeds the steal roll in Ball, so episodes replay exactly
        self._place(sp.blue_pos, sp.red_pos, sp.ball_pos, sp.holder)
        self._step = 0
        return self._state(), self._info([])

    def step(self, actions: dict[str, Any]) -> tuple[dict, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("call reset() before step(); there is no auto-reset (ICD §4)")
        move, triggers = parse_actions(actions, self.n_blue)
        # Start-of-step rule (ICD §3): kicks use possession at the start of the block.
        for player, trig in zip(self._game.blues, triggers):
            if trig == 1 and self._game.ball.possession_player is player:
                raise NotImplementedError(
                    "the PyBullet adapter doesn't implement kicking yet (deferred with kick_enabled, #21); "
                    "the RL side sends kick_trigger = 0 at milestone 1"
                )

        _, _, terminated, _, _ = self._game.step(move * self.cfg.max_speed)
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
        """Disconnects this env's own pybullet simulation; other envs keep running."""
        self._ready = False
        self._game.close()

    # --- Test hook ---------------------------------------------------------------
    def set_state_for_testing(
        self,
        *,
        blue_pos: Any = None,
        red_pos: Any = None,
        ball_pos: Any = None,
        holder: Any = _UNSET,
        step: int | None = None,
    ) -> dict:
        """Place bodies directly (velocities zeroed), like ``MockEnv.set_state_for_testing``. For unit
        tests only, after a ``reset``. It rebuilds the simulation, so contact history starts over.
        ``holder`` is None (a loose ball) or ``("blue", i)``; a blue holder carries the ball."""
        s = self._state()
        blue = blue_pos if blue_pos is not None else [b["pos"] for b in s["blue"]]
        red = red_pos if red_pos is not None else [r["pos"] for r in s["red"]]
        ball = ball_pos if ball_pos is not None else s["ball"]["pos"]
        if holder is _UNSET:
            holder = next((("blue", b["id"]) for b in s["blue"] if b["has_possession"]), None)
        if holder is not None and holder[0] != "blue":
            raise NotImplementedError("the PyBullet test hook places a loose ball or a blue holder only")
        self._place(blue, red, ball, None if holder is None else holder[1])
        if step is not None:
            self._step = step
        return self._state()

    # --- ICD plumbing -------------------------------------------------------------
    def _place(self, blue_pos: Any, red_pos: Any, ball_pos: Any, holder: int | None) -> None:
        """Rebuild the game with these bodies at rest."""
        options = {
            "blue_positions": [tuple(map(float, xy)) for xy in np.asarray(blue_pos, dtype=float).reshape(self.n_blue, 2)],
            "red_positions": [tuple(map(float, xy)) for xy in np.asarray(red_pos, dtype=float).reshape(self.n_red, 2)],
            "ball_position": tuple(map(float, np.asarray(ball_pos, dtype=float))),
        }
        if holder is not None:
            options["ball_holder"] = holder
        self._game.reset(seed=self._seed, options=options)
        self._ready = True

    def _holder_id(self, player: Any) -> tuple[str, int] | None:
        if player is None:
            return None
        for team, players in (("blue", self._game.blues), ("red", self._game.reds)):
            for i, q in enumerate(players):
                if q is player:
                    return (team, i)
        raise RuntimeError("the ball is held by a body that isn't a player in this game")

    def _state(self) -> dict:
        game = self._game
        holder = game.ball.possession_player

        def xy(v: np.ndarray) -> list[float]:
            return [float(v[0]), float(v[1])]

        def entity(i: int, player: Any) -> dict:
            return {"id": i, "pos": xy(player.position), "vel": xy(player.velocity), "has_possession": holder is player}

        return {
            "step": self._step,
            "blue": [entity(i, b) for i, b in enumerate(game.blues)],
            "red": [entity(j, r) for j, r in enumerate(game.reds)],
            "ball": {"pos": xy(game.ball.position), "vel": xy(game.ball.velocity)},
        }

    def _info(self, events: list[dict]) -> dict:
        return {"events": events, "kick_applied": [False] * self.n_blue}
### END AI GENERATED
