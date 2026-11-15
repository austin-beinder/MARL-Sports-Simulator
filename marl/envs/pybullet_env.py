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
#   - 2026-11-07: Kicking (ICD §6): kick_vec -> the ball's velocity, kick
#     events and kick_applied; max_kick_speed and ball_damping passed through.
#   - 2026-11-08: free_speed_factor passed through (blue's speed without the
#     ball).
#   - 2026-11-08: The passing_drill scenario: keep-away with the blues frozen
#     in place (their move actions are ignored), so only kicks matter.
#   - 2026-11-08: reset raises if a passing_drill episode would start without
#     a blue holding the ball.
#   - 2026-11-09: The pass_practice scenario: frozen blues, no red, and the
#     episode ends once the one kick has played out.
#   - 2026-11-09: pass_practice_moving: the same, with the blues free to move;
#     a teammate who can still run to the ball in time keeps the kick alive.
#   - 2026-11-11: pass_practice_moving no longer ends when a kick plays out:
#     it runs to the time limit, with as many kicks as the blues like.
#     (Austin Beinder)
#   - 2026-11-13: s["ball"]["passer"] (ICD §5): the blue whose kick the
#     holder picked up. (Austin Beinder)
#   - 2026-11-13: The optional "call" action, echoed as each blue's "calling"
#     in s. (Austin Beinder)
#   - 2026-11-14: The claim_ball scenario: a loose ball, no red, terminated
#     when a blue picks it up. (Austin Beinder)
#   - 2026-11-14: In the GUI, a blue who calls turns a brighter blue
#     (Player.show_calling). (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters; no
#     behaviour change. (Austin Beinder)
# ==============================================================================
### AI GENERATED
"""ICD adapter around the PyBullet keep-away game (``pybulletGame.KeepAwayEnv``).

It has the same ``reset(seed, options)`` / ``step(actions)`` interface as ``MockEnv``
and translates in both directions:

* actions: ICD ``move`` in [-1, 1] (2-norm capped) x ``max_speed`` -> each blue's velocity in m/s;
  ``kick_vec`` -> the ball's velocity ``max_kick_speed x kick_vec / max(1, |kick_vec|)`` (ICD §6);
* state: the game's players and ball -> ICD ``s`` (``step``, ``blue``, ``red``, ``ball``);
* rewards are dropped: the RL layer computes its own (ICD §4);
* spawns come from ``sample_spawn``, like the mock, so both envs start from the same states;
* red possession ends the episode on the physics tick it happens (``red_win_time`` = one tick),
  and blue's "hold for 20 s" win is switched off: the ICD has no blue-win terminal (§7);
* only an opponent can take the ball from its carrier (teammate handoff off, ICD §8);
* goto_target: no red, and the ball is an immobile target nobody can touch or hold (§9).
* passing_drill (2026-11-08): keep-away rules, but the blues can't move (their ``move`` is
  ignored), so the only way to keep the ball from red is to pass it (§9).
* pass_practice (2026-11-09): frozen blues as in passing_drill, no red, and one kick per episode:
  the episode terminates once a kick has played out, i.e. some blue holds the ball again (a
  teammate caught it, or the kicker got it back) or the loose ball can't reach a teammate any
  more: what it has left to roll (``ln(1 + |v|) / ball_damping``) is less than the gap to the
  nearest teammate. A slow ball crawls for seconds, so waiting for it to stop would mostly hit
  the time limit (§7, §9).
* pass_practice_moving (2026-11-09): pass_practice with the blues free to move and, since
  2026-11-11, no one-kick limit: the episode runs to the time limit (``truncated``), and the
  blues can pass as often as they like. After a miss somebody has to fetch the loose ball.

Possession is contact-based in the sim (player edge touching ball edge, or an opponent touching
the carrier), not ``possession_radius``. A kicker can touch the ball again at once (no cooldown).

Each instance has its own pybullet simulation (``KeepAwayEnv`` opens a connection per game), so
any number can be mid-episode in one process. Only one can render (pybullet allows one GUI).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# AI GENERATED (drill sets)
from marl.common.config import FROZEN_SCENARIOS, PASS_PRACTICE_SCENARIOS, SharedConfig
from marl.common.spawn import RESET_OPTION_KEYS, sample_spawn
from marl.envs.mock_env import parse_actions, parse_calls  # AI GENERATED (2026-11-13: parse_calls)
from pybulletGame import Entity, KeepAwayConfig, KeepAwayEnv

_UNSET = object()


class PyBulletEnv:
    """ICD-conformant wrapper around ``KeepAwayEnv``. ``reset`` and ``step`` speak plain Python types."""

    def __init__(self, cfg: SharedConfig, render_mode: str | None = None, print_collisions: bool = False):
        """render_mode="human" opens the pybullet GUI (only if this process has no pybullet connection
        yet). print_collisions turns on the sim's collision debug prints, for the whole process."""
        # AI GENERATED (drills)
        if cfg.scenario not in ("keepaway", "goto_target", "passing_drill",
                                *PASS_PRACTICE_SCENARIOS,
                                "claim_ball"):  # AI GENERATED (2026-11-14)
            raise NotImplementedError(f"the PyBullet env doesn't implement scenario {cfg.scenario!r}")
        self.cfg = cfg
        self.n_blue, self.n_red = cfg.n_blue, cfg.n_red
        # AI GENERATED (2026-11-08): blues stand still
        self._frozen = cfg.scenario in FROZEN_SCENARIOS
        # AI GENERATED (2026-11-09; frozen only 2026-11-11): ends when the kick plays out
        self._one_kick = cfg.scenario == "pass_practice"
        # AI GENERATED (2026-11-09): who kicked this episode (pass_practice)
        self._kicker: int | None = None
        ### AI GENERATED (2026-11-13: s["ball"]["passer"], see _track_passer)
        self._kicked_by: int | None = None  # the blue whose kick the loose ball is from
        self._passer: int | None = None
        self._calls = [False] * cfg.n_blue  # each blue's latest call, echoed in s (ICD §5)
        ### END AI GENERATED
        Entity.print_collisions = print_collisions
        episode_time = cfg.max_episode_steps * cfg.control_k * cfg.physics_dt
        self._game = KeepAwayEnv(
            KeepAwayConfig(
                blue_max_speed=cfg.max_speed,
                red_max_speed=cfg.red_max_speed,
                possession_speed_multiplier=cfg.carry_speed_factor,
                free_speed_multiplier=cfg.free_speed_factor,  # 2026-11-08
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
                max_kick_speed=cfg.max_kick_speed,  # also the ball's top speed
                ball_damping=cfg.ball_damping,
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
        s = self._state()
        # AI GENERATED (2026-11-08)
        if ((self._frozen or self.cfg.scenario in PASS_PRACTICE_SCENARIOS)
                and not any(b["has_possession"] for b in s["blue"])):
            # AI GENERATED
            raise RuntimeError(
                f"a {self.cfg.scenario} episode must start with a blue holding the ball"
            )
        return s, self._info([])

    def step(self, actions: dict[str, Any]) -> tuple[dict, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("call reset() before step(); there is no auto-reset (ICD §4)")
        move, triggers, kick_vecs = parse_actions(actions, self.n_blue)
        # AI GENERATED (2026-11-13): echoed in s (ICD §5)
        self._calls = parse_calls(actions, self.n_blue)
        # AI GENERATED (2026-11-14): callers turn a brighter blue in the GUI
        if self._game.render_mode == "human":
            for blue, calling in zip(self._game.blues, self._calls):
                blue.show_calling(calling)
        # AI GENERATED (2026-11-08): passing_drill ignores move (still validated above)
        if self._frozen:
            move = np.zeros_like(move)
        # ICD §6: release velocity = max_kick_speed * kick_vec / max(1, |kick_vec|). The game
        # applies a kick only if that player holds the ball at the start of the step (ICD §3).
        kicks = [
            self.cfg.max_kick_speed * kv / max(1.0, float(np.linalg.norm(kv)))
            if trig == 1 else None
            for trig, kv in zip(triggers, kick_vecs)
        ]

        _, _, terminated, _, game_info = self._game.step(move * self.cfg.max_speed, kicks=kicks)
        self._step += 1
        ### AI GENERATED (2026-11-09: pass_practice ends once its kick has played out)
        if self._one_kick:
            if any(game_info["kicks"]):
                self._kicker = [bool(k) for k in game_info["kicks"]].index(True)
            if self._kicker is not None and not terminated:
                terminated = self._kick_played_out()
        ### END AI GENERATED
        ### AI GENERATED (2026-11-14: claim_ball ends when a blue gets the ball)
        if self.cfg.scenario == "claim_ball" and self._game.ball.possession_player is not None:
            terminated = True
        ### END AI GENERATED
        truncated = (not terminated) and self._step >= self.cfg.max_episode_steps
        if terminated or truncated:
            self._ready = False  # no auto-reset; the returned state is the true final state
        kick_applied = [bool(k) for k in game_info["kicks"]]
        # The kick happens before the first physics tick, so its event comes first
        events = [{"type": "kick", "agent": ("blue", i)}
                  for i, kicked in enumerate(kick_applied) if kicked]
        events += [
            {"type": "possession_change", "from": self._holder_id(old), "to": self._holder_id(new)}
            for _, old, new in self._game.possession_events
        ]
        self._track_passer(events)  # AI GENERATED (2026-11-13)
        return self._state(), bool(terminated), truncated, self._info(events, kick_applied)

    ### AI GENERATED (2026-11-13)
    def _track_passer(self, events: list[dict]) -> None:
        """s["ball"]["passer"]: the blue who passed the ball to its holder. A blue who picks up a
        teammate's kick was passed to by that teammate; one who picks its own kick back up keeps
        the passer it had; any other pickup (a loose ball nobody kicked, a red) clears it. Every
        kick counts, however short: the RL side's pass rules (minimum distance) don't apply."""
        for event in events:
            if event["type"] == "kick":
                self._kicked_by = event["agent"][1]
            elif event["type"] == "possession_change" and event["to"] is not None:
                team, i = event["to"]
                if team != "blue" or self._kicked_by is None:
                    self._passer = None
                elif self._kicked_by != i:
                    self._passer = self._kicked_by
                self._kicked_by = None
    ### END AI GENERATED

    ### AI GENERATED (2026-11-09)
    def _kick_played_out(self) -> bool:
        """pass_practice: a blue holds the ball again, or the loose ball can't roll as far as a
        (frozen) teammate any more. (pass_practice_moving runs to the time limit instead,
        2026-11-11.)"""
        ball = self._game.ball
        if ball.possession_player is not None:
            return True
        if self.cfg.ball_damping <= 0:
            return False  # an undamped ball rolls on: only a catch or the time limit ends it
        speed = float(np.linalg.norm(np.asarray(ball.velocity, dtype=float)[:2]))
        roll_left = np.log1p(speed) / self.cfg.ball_damping
        pos = np.asarray(ball.position, dtype=float)[:2]
        gap = min(np.linalg.norm(np.asarray(b.position, dtype=float)[:2] - pos)
                  - self.cfg.player_radius - ball.radius
                  for i, b in enumerate(self._game.blues) if i != self._kicker)
        return roll_left < gap
    ### END AI GENERATED

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
        # AI GENERATED (2026-11-09): a fresh pass_practice episode has its kick to come
        self._kicker = None
        self._kicked_by = self._passer = None  # AI GENERATED (2026-11-13): nobody has passed yet
        self._calls = [False] * self.n_blue  # AI GENERATED (2026-11-13): nor called

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
            # AI GENERATED (2026-11-13: ICD §5)
            "blue": [{**entity(i, b), "calling": self._calls[i]}
                     for i, b in enumerate(game.blues)],
            "red": [entity(j, r) for j, r in enumerate(game.reds)],
            "ball": {"pos": xy(game.ball.position), "vel": xy(game.ball.velocity),
                     "passer": self._passer},  # AI GENERATED (2026-11-13: ICD §5)
        }

    def _info(self, events: list[dict], kick_applied: list[bool] | None = None) -> dict:
        return {"events": events, "kick_applied": kick_applied or [False] * self.n_blue}
### END AI GENERATED
