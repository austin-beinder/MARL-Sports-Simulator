# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-18: Moved action validation into module-level parse_actions()
#     so the PyBullet adapter (pybullet_env.py) shares it. (Austin Beinder)
#   - 2026-11-07: parse_actions also returns the kick vectors (the PyBullet
#     adapter now kicks). The mock stays movement-only. (Austin Beinder)
#   - 2026-11-08: Blue's speed without the ball is free_speed_factor x
#     max_speed. (Austin Beinder)
#   - 2026-11-08: Refuses the passing_drill scenario (it needs kicks).
#     (Austin Beinder)
#   - 2026-11-09: Refuses pass_practice and pass_practice_moving too. (Austin Beinder)
#   - 2026-11-13: s["ball"]["passer"] (ICD §5), always None: the mock can't
#     kick, so nobody ever passes. (Austin Beinder)
#   - 2026-11-13: The optional "call" action, echoed as each blue's "calling"
#     in s (parse_calls). (Austin Beinder)
#   - 2026-11-23: Refuses the soccer scenario (no goals). (Austin Beinder)
#   - 2026-11-14: The claim_ball scenario: a loose ball, no red, terminated
#     when a blue picks it up. (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters and set
#     _calls in __init__ (attribute-defined-outside-init); no behaviour
#     change. (Austin Beinder)
#   - 2026-10-10: Generated this entire file: a kinematic 2D mock env that
#     implements docs/ICD.md v0.2 (action repeat, per-tick termination, walls,
#     possession radius, carry slowdown, pursuit red, goto_target scenario,
#     strict action validation), following design #8, #16, #19, #22 and
#     Phase 3 decision H.
# ==============================================================================
"""Kinematic 2D mock of the keep-away env, implementing ``docs/ICD.md`` (v0.2).

Dynamics (Phase 3 decision H), per physics tick:

* blue velocity = the 2-norm-capped ``move`` x current max speed (carry slowdown
  if holding the ball), set instantly;
* red moves straight at the ball at ``red_max_speed`` (never overshooting it);
* position += velocity * dt, clamped at the walls; the velocity component into
  a wall is zeroed;
* a possessed ball follows its carrier; a loose ball stays at rest;
* possession is checked after every tick; red possession ends the block early
  with ``terminated = True`` (ICD §3, §8).

Expected mock-vs-sim differences (report them): no player-player collisions;
no ball physics (kicking is PyBullet-only, see ``step``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from marl.common.config import SharedConfig
from marl.common.spawn import RESET_OPTION_KEYS, sample_spawn

Holder = tuple[str, int] | None
_UNSET = object()


class MockEnv:
    """ICD-conformant mock env. ``reset`` and ``step`` speak plain Python types."""

    def __init__(self, cfg: SharedConfig):
        # AI GENERATED (2026-11-08)
        if cfg.scenario in ("passing_drill", "pass_practice", "pass_practice_moving"):
            raise NotImplementedError(
                f"the mock env can't run {cfg.scenario}: it has no kicks (use the PyBullet env)"
            )
        # AI GENERATED (2026-11-23): no goals in the mock
        if cfg.scenario == "soccer":
            raise NotImplementedError("the mock env can't run soccer: it has no goals (use the PyBullet env)")
        self.cfg = cfg
        self.n_blue, self.n_red = cfg.n_blue, cfg.n_red
        half = np.asarray(cfg.field_size, dtype=float) / 2.0
        self._lo, self._hi = -half + cfg.player_radius, half - cfg.player_radius
        # goto_target: ball is an immobile target  # AI GENERATED (claim_ball)
        self._possession = cfg.scenario in ("keepaway", "claim_ball")
        self._ready = False
        # AI GENERATED (2026-11-15): nobody calls before the first step (reset sets it again)
        self._calls: list[bool] = [False] * self.n_blue

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
        self._blue_pos = np.array(sp.blue_pos, dtype=float).reshape(self.n_blue, 2)
        self._red_pos = np.array(sp.red_pos, dtype=float).reshape(self.n_red, 2)
        self._ball_pos = np.array(sp.ball_pos, dtype=float)
        self._blue_vel = np.zeros_like(self._blue_pos)
        self._red_vel = np.zeros_like(self._red_pos)
        self._ball_vel = np.zeros(2)
        self._holder: Holder = ("blue", sp.holder) if sp.holder is not None else None
        self._step = 0
        self._calls = [False] * self.n_blue  # AI GENERATED (2026-11-13): nobody calls at reset
        self._ready = True
        return self._state(), self._info([])

    def step(self, actions: dict[str, Any]) -> tuple[dict, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("call reset() before step(); there is no auto-reset (ICD §4)")
        move, triggers, _ = self._parse_actions(actions)
        # AI GENERATED (2026-11-13): echoed in s (ICD §5)
        self._calls = parse_calls(actions, self.n_blue)

        # Start-of-step rule (ICD §3): kicks use possession at the start of the block.
        for i, trig in enumerate(triggers):
            if trig == 1 and self._holder == ("blue", i):
                ### AI GENERATED (2026-11-07: kicking is PyBullet-only)
                raise NotImplementedError(
                    "the mock env doesn't implement kicking (it has no ball physics); "
                    "train with kicks on run.env = 'pybullet'"
                )
                ### END AI GENERATED

        events: list[dict] = []
        terminated = False
        for _ in range(self.cfg.control_k):
            self._tick(move)
            event = self._update_possession()
            if event is not None:
                events.append(event)
            if self._holder is not None and self._holder[0] == "red":
                terminated = True  # mid-block termination (ICD §3): stop the block now
                break
            ### AI GENERATED (2026-11-14: claim_ball ends when a blue gets the ball)
            if self._holder is not None and self.cfg.scenario == "claim_ball":
                terminated = True
                break
            ### END AI GENERATED
        self._step += 1
        truncated = (not terminated) and self._step >= self.cfg.max_episode_steps
        if terminated or truncated:
            self._ready = False  # no auto-reset; the returned state is the true final state
        return self._state(), terminated, truncated, self._info(events)

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
        """Place bodies directly (velocities zeroed). For unit tests only."""
        if blue_pos is not None:
            self._blue_pos = np.array(blue_pos, dtype=float).reshape(self.n_blue, 2)
        if red_pos is not None:
            self._red_pos = np.array(red_pos, dtype=float).reshape(self.n_red, 2)
        if ball_pos is not None:
            self._ball_pos = np.array(ball_pos, dtype=float)
        if holder is not _UNSET:
            self._holder = holder
        if step is not None:
            self._step = step
        if self._holder is not None and self._holder[0] == "blue":
            self._ball_pos = self._blue_pos[self._holder[1]].copy()
        self._blue_vel = np.zeros((self.n_blue, 2))
        self._red_vel = np.zeros((self.n_red, 2))
        self._ball_vel = np.zeros(2)
        self._ready = True
        return self._state()

    # --- Physics -----------------------------------------------------------------
    def _tick(self, move: np.ndarray) -> None:
        cfg, dt = self.cfg, self.cfg.physics_dt
        # AI GENERATED (free_speed_factor)
        speed = np.full(self.n_blue, cfg.max_speed * cfg.free_speed_factor)
        if self._holder is not None and self._holder[0] == "blue":
            # AI GENERATED (was *=, now not stacked)
            speed[self._holder[1]] = cfg.max_speed * cfg.carry_speed_factor
        blue_vel = move * speed[:, None]

        red_vel = np.zeros((self.n_red, 2))
        if self.n_red:
            to_ball = self._ball_pos - self._red_pos
            dist = np.linalg.norm(to_ball, axis=1)
            # Head for the ball at full speed, but don't overshoot it within one tick.
            mag = np.minimum(cfg.red_max_speed, dist / dt)
            safe = np.where(dist > 1e-12, dist, 1.0)
            red_vel = to_ball / safe[:, None] * mag[:, None]

        self._blue_pos, self._blue_vel = self._integrate(self._blue_pos, blue_vel, dt)
        self._red_pos, self._red_vel = self._integrate(self._red_pos, red_vel, dt)
        if self._holder is not None and self._holder[0] == "blue":
            i = self._holder[1]
            self._ball_pos = self._blue_pos[i].copy()
            self._ball_vel = self._blue_vel[i].copy()

    def _integrate(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        new = pos + vel * dt
        clipped = np.clip(new, self._lo, self._hi)
        vel = np.where(clipped != new, 0.0, vel)  # zero the into-wall component
        return clipped, vel

    def _update_possession(self) -> dict | None:
        if not self._possession:
            return None
        r = self.cfg.possession_radius
        prev = self._holder
        if prev is not None and prev[0] == "blue":
            # Only red can take the ball from a blue carrier (ICD §8).
            if not self.n_red:
                return None
            d = np.linalg.norm(self._red_pos - self._ball_pos, axis=1)
            j = int(np.argmin(d))
            if d[j] > r:
                return None
            new: Holder = ("red", j)
        elif prev is None:
            d = np.linalg.norm(np.concatenate([self._blue_pos, self._red_pos]) - self._ball_pos, axis=1)
            k = int(np.argmin(d))  # closest wins; ties go to the lower index (blue first)
            if d[k] > r:
                return None
            new = ("blue", k) if k < self.n_blue else ("red", k - self.n_blue)
        else:
            return None  # red holds the ball; the episode is already over
        self._holder = new
        if new[0] == "blue":
            self._ball_pos = self._blue_pos[new[1]].copy()
            self._ball_vel = self._blue_vel[new[1]].copy()
        return {"type": "possession_change", "from": prev, "to": new}

    # --- ICD plumbing -------------------------------------------------------------
    # AI GENERATED (kick vectors)
    def _parse_actions(self, actions: Any) -> tuple[np.ndarray, list[int], np.ndarray]:
        ### AI GENERATED
        return parse_actions(actions, self.n_blue)
        ### END AI GENERATED

    def _state(self) -> dict:
        def entity(team: str, i: int, pos: np.ndarray, vel: np.ndarray) -> dict:
            e = {
                "id": i,
                "pos": [float(pos[0]), float(pos[1])],
                "vel": [float(vel[0]), float(vel[1])],
                "has_possession": self._holder == (team, i),
            }
            if team == "blue":  # AI GENERATED (2026-11-13): the blue's latest call (ICD §5)
                e["calling"] = bool(getattr(self, "_calls", [False] * self.n_blue)[i])
            return e

        return {
            "step": self._step,
            "blue": [entity("blue", i, self._blue_pos[i], self._blue_vel[i]) for i in range(self.n_blue)],
            "red": [entity("red", j, self._red_pos[j], self._red_vel[j]) for j in range(self.n_red)],
            "ball": {
                "pos": [float(self._ball_pos[0]), float(self._ball_pos[1])],
                "vel": [float(self._ball_vel[0]), float(self._ball_vel[1])],
                "passer": None,  # AI GENERATED (2026-11-13): no kicks, so no passes (ICD §5)
            },
        }

    def _info(self, events: list[dict]) -> dict:
        return {"events": events, "kick_applied": [False] * self.n_blue}


### AI GENERATED (moved out of MockEnv so the PyBullet adapter can share it)
def parse_actions(actions: Any, n_blue: int) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Validate actions strictly (ICD §6); return capped move vectors, triggers and raw kick
    vectors [n_blue, 2] (the env applies the kick speed cap, ICD §6).

    Shared by every ICD env (the mock and the PyBullet adapter) so they reject the same inputs.
    """
    if not isinstance(actions, dict):
        raise TypeError(f"actions must be a dict, got {type(actions).__name__}")
    if set(actions) != {"blue"}:
        raise ValueError(f"actions must have exactly the key 'blue', got {sorted(actions)}")
    blue = actions["blue"]
    if not isinstance(blue, list):
        raise TypeError(f"actions['blue'] must be a list, got {type(blue).__name__}")
    if len(blue) != n_blue:
        raise ValueError(f"actions['blue'] must have {n_blue} entries, got {len(blue)}")
    move = np.zeros((n_blue, 2))
    kick_vecs = np.zeros((n_blue, 2))  # 2026-11-07
    triggers = []
    for i, a in enumerate(blue):
        where = f"actions['blue'][{i}]"
        if not isinstance(a, dict):
            raise TypeError(f"{where} must be a dict")
        # AI GENERATED (call)
        if not ({"move", "kick_trigger", "kick_vec"}
                <= set(a) <= {"move", "kick_trigger", "kick_vec", "call"}):
            raise ValueError(
                f"{where} must have keys move, kick_trigger, kick_vec (and optionally call); "
                f"got {sorted(a)}"
            )
        ### AI GENERATED (2026-11-13: the optional call, ICD §6)
        if "call" in a and (type(a["call"]) is not int or a["call"] not in (0, 1)):
            raise ValueError(f"{where}['call'] must be an int 0 or 1, got {a['call']!r}")
        ### END AI GENERATED
        move[i] = _vec2(a["move"], f"{where}['move']")
        kick_vecs[i] = _vec2(a["kick_vec"], f"{where}['kick_vec']")
        trig = a["kick_trigger"]
        if type(trig) is not int:
            raise TypeError(f"{where}['kick_trigger'] must be an int 0 or 1, got {trig!r}")
        if trig not in (0, 1):
            raise ValueError(f"{where}['kick_trigger'] must be 0 or 1, got {trig}")
        triggers.append(trig)
    norm = np.linalg.norm(move, axis=1, keepdims=True)
    move = np.where(norm > 1.0, move / np.maximum(norm, 1e-12), move)  # 2-norm speed cap (ICD §6)
    return move, triggers, kick_vecs


def parse_calls(actions: dict, n_blue: int) -> list[bool]:  # AI GENERATED (2026-11-13)
    """Each blue's call (ICD §6, optional, default 0) from actions ``parse_actions`` accepted."""
    return [bool(a.get("call", 0)) for a in actions["blue"][:n_blue]]
### END AI GENERATED


def _vec2(v: Any, where: str) -> tuple[float, float]:
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"{where} must be a list of 2 numbers, got {v!r}")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"{where} entries must be Python floats, got {type(x).__name__}")
        if not math.isfinite(x) or not -1.0 <= x <= 1.0:
            raise ValueError(f"{where} entries must be finite and in [-1, 1], got {x!r}")
    return float(v[0]), float(v[1])
