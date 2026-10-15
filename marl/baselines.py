# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: scripted baseline policies
#     (random, stand-still, a lookahead flee heuristic) and the goto_target
#     oracle, all usable through the shared evaluation protocol (design #22,
#     #23).
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
"""Scripted baseline policies (design #23) and the goto_target oracle.

Each baseline is a callable ``policy(s) -> actions`` on the global state s, so it
runs through exactly the same protocol as the learned policy
(``marl.rl.evaluate.run_episodes``: same env, same fixed seeds, same metrics).

* ``random``: each move component uniform in [-1, 1], every step (seeded).
* ``still``: move = [0, 0] always.
* ``flee``: the scripted reference for keep-away, and the policy the red speed
  is calibrated against (#22):
    - ball loose: run straight at it at full speed;
    - I hold it: try ``n_headings`` directions and pick the one whose
      straight-line lookahead (``horizon_s`` seconds) keeps the ball farthest
      from the nearest red over the whole lookahead. The lookahead models the
      walls (my motion is clamped) and red's rule (it runs straight at the ball);
    - a teammate holds it: stand still (there is no passing yet).
* ``oracle``: goto_target only. Drive straight at the target and land on it
  (an approximate upper bound for that task).

Scripted policies read the global state directly. Under milestone-1 full
observability the learned policy has the same information, so the comparison is fair.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from marl.common.config import SharedConfig
from marl.rl.state import parse_state

Policy = Callable[[dict], dict]
BASELINES = ("random", "still", "flee", "oracle")


def baselines_for(scenario: str) -> tuple[str, ...]:
    return ("random", "still", "flee") if scenario == "keepaway" else ("random", "still", "oracle")


def make_baseline(name: str, cfg: SharedConfig, seed: int = 0) -> Policy:
    if name == "random":
        return RandomPolicy(cfg.n_blue, seed)
    if name == "still":
        return StillPolicy(cfg.n_blue)
    if name == "flee":
        if cfg.scenario != "keepaway":
            raise ValueError("flee is a keep-away baseline")
        return FleePolicy(cfg)
    if name == "oracle":
        if cfg.scenario != "goto_target":
            raise ValueError("the oracle exists for goto_target only")
        return OraclePolicy(cfg)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")


def _actions(moves: np.ndarray) -> dict:
    return {"blue": [{"move": [float(x), float(y)], "kick_trigger": 0, "kick_vec": [0.0, 0.0]} for x, y in moves]}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(2)


class RandomPolicy:
    def __init__(self, n_blue: int, seed: int = 0):
        self.n_blue = n_blue
        self.rng = np.random.default_rng(seed)

    def __call__(self, s: dict) -> dict:
        return _actions(self.rng.uniform(-1.0, 1.0, size=(self.n_blue, 2)))


class StillPolicy:
    def __init__(self, n_blue: int):
        self.n_blue = n_blue

    def __call__(self, s: dict) -> dict:
        return _actions(np.zeros((self.n_blue, 2)))


class OraclePolicy:
    """goto_target: full speed at the target, scaled down on the last step to land on it."""

    def __init__(self, cfg: SharedConfig):
        self.cfg = cfg
        self.step_len = cfg.max_speed * cfg.dt_control  # meters per control step at full speed

    def __call__(self, s: dict) -> dict:
        st = parse_state(s, self.cfg.n_blue, self.cfg.n_red)
        to = st.ball_pos - st.blue_pos[0]
        dist = float(np.linalg.norm(to))
        move = to * (min(1.0, dist / self.step_len) / dist) if dist > 1e-12 else np.zeros(2)
        return _actions(move[None, :])


class FleePolicy:
    def __init__(self, cfg: SharedConfig, n_headings: int = 16, horizon_s: float = 1.0, substep_s: float = 0.1):
        self.cfg = cfg
        half = np.asarray(cfg.field_size, dtype=float) / 2.0
        self.lo, self.hi = -half + cfg.player_radius, half - cfg.player_radius
        angles = 2.0 * np.pi * np.arange(n_headings) / n_headings
        self.dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # [K, 2] unit headings
        self.n_sub = max(1, round(horizon_s / substep_s))
        self.dt = horizon_s / self.n_sub
        self.v_carry = cfg.max_speed * cfg.carry_speed_factor

    def __call__(self, s: dict) -> dict:
        st = parse_state(s, self.cfg.n_blue, self.cfg.n_red)
        moves = np.zeros((self.cfg.n_blue, 2))
        holders = np.flatnonzero(st.blue_poss)
        if len(holders):
            i = int(holders[0])
            moves[i] = self.flee_heading(st.blue_pos[i], st.red_pos)
        elif not st.ball_held:
            for i in range(self.cfg.n_blue):
                moves[i] = _unit(st.ball_pos - st.blue_pos[i])
        return _actions(moves)

    def flee_heading(self, me: np.ndarray, reds: np.ndarray) -> np.ndarray:
        """The heading whose lookahead maximizes the minimum ball-to-red distance."""
        if len(reds) == 0:
            return np.zeros(2)
        K = len(self.dirs)
        p = np.repeat(me[None, :], K, axis=0)  # my (= the ball's) position per heading [K, 2]
        r = np.repeat(reds[None, :, :], K, axis=0)  # red positions per heading [K, M, 2]
        closest = np.full(K, np.inf)
        for _ in range(self.n_sub):
            p = np.clip(p + self.dirs * (self.v_carry * self.dt), self.lo, self.hi)  # walls stop me
            to = p[:, None, :] - r
            d = np.linalg.norm(to, axis=-1)
            step = np.minimum(self.cfg.red_max_speed * self.dt, d)  # red runs at the ball, no overshoot
            r = r + to / np.maximum(d, 1e-12)[..., None] * step[..., None]
            closest = np.minimum(closest, np.linalg.norm(p[:, None, :] - r, axis=-1).min(axis=1))
        return self.dirs[int(np.argmax(closest))]
