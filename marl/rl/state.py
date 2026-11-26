# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: strict parsing of the ICD global
#     state s and the explicit flattening spec for the critic's global input
#     (design #11). Changed the flattener to scale ALL velocities (incl. red) by
#     max_speed, a student-approved amendment to #11.
#   - 2026-11-07: state_blocks, the flattening spec as blocks, so a warm start
#     can grow the critic's first layer when the team sizes change.
#     s["ball"]["passer"] (optional): who passed the ball to its holder, and
#     with rl.passer_obs a per-blue "passed it" flag; each blue's optional
#     "calling" (its latest call), and with rl.call_enabled a per-blue call
#     flag. Pylint cleanup: wrapped lines over 100 characters; no behaviour
#     change. (Austin Beinder)
#   - 2026-11-25: Citation cleanup (backlog task 9): one Usage entry per PR,
#     oldest first; no behaviour change.
# ==============================================================================
"""Strict parsing of the global state ``s`` (ICD §5) and the flattening spec (#11).

``parse_state`` checks the structure EXACTLY: an unknown key raises, so hidden
env state can't be silently ignored (ICD §5 "hidden state rule"). Both the
flattener and the obs builder (``marl/rl/obs.py``) work on its output.

Flattening spec (the critic's global input; ``horizon_mode = infinite``):

    for each blue (by id), then each red (by id):
        pos / half_field (2), vel / max_speed (2), has_possession as 0/1 (1)
    ball: pos / half_field (2), vel / max_speed (2)

Every velocity uses ONE unit, blue's ``max_speed`` (amendment to #11, approved
2026-10-11). ``red_max_speed`` is the calibration knob (#22); if it were a
divisor, recalibrating red would silently change what the inputs mean.
``step`` and ``id`` are dropped (no time anywhere in infinite mode; the id is
implied by the slot). Dimension = 5 * (N + M) + 4.

With ``rl.passer_obs`` (2026-11-13) each blue's entry gets a 6th number: 1 if that blue passed the
ball to its current (or latest) holder, ``s["ball"]["passer"]``. Dimension = 6 N + 5 M + 4.
``s["ball"]["passer"]`` is optional: a state without it has no passer.

With ``rl.call_enabled`` (2026-11-13) each blue's entry gets one more number after that: 1 while
the blue is calling (``s["blue"][i]["calling"]``, its call on the latest step). Optional too: a
blue without it isn't calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl.common.config import SharedConfig

_TOP_KEYS = frozenset({"step", "blue", "red", "ball"})
_ENTITY_KEYS = frozenset({"id", "pos", "vel", "has_possession"})
_BALL_KEYS = frozenset({"pos", "vel"})
# AI GENERATED (2026-11-13): hand-built states may leave it out
_BALL_OPTIONAL = frozenset({"passer"})
_BLUE_OPTIONAL = frozenset({"calling"})  # AI GENERATED (2026-11-13): likewise


@dataclass(frozen=True)
class StateArrays:
    """The global state as numpy arrays (SI units, world frame)."""

    blue_pos: np.ndarray  # [N, 2]
    blue_vel: np.ndarray  # [N, 2]
    blue_poss: np.ndarray  # [N] bool
    red_pos: np.ndarray  # [M, 2]
    red_vel: np.ndarray  # [M, 2]
    red_poss: np.ndarray  # [M] bool
    ball_pos: np.ndarray  # [2]
    ball_vel: np.ndarray  # [2]
    # AI GENERATED (2026-11-13): the blue who passed the ball to its holder
    passer: int | None = None
    # AI GENERATED (2026-11-13): [N] bool, None = nobody calls
    blue_calling: np.ndarray | None = None

    @property
    def ball_held(self) -> bool:
        return bool(self.blue_poss.any() or self.red_poss.any())


def parse_state(s: Any, n_blue: int, n_red: int) -> StateArrays:
    """Validate ``s`` against ICD §5 and convert it to arrays."""
    if not isinstance(s, dict):
        raise TypeError(f"state must be a dict, got {type(s).__name__}")
    _check_keys(s, _TOP_KEYS, "s")
    # AI GENERATED (2026-11-13: calling)
    blue = _entities(s["blue"], n_blue, "s['blue']", _BLUE_OPTIONAL)
    red = _entities(s["red"], n_red, "s['red']")
    ball = s["ball"]
    if not isinstance(ball, dict):
        raise TypeError("s['ball'] must be a dict")
    ### AI GENERATED (2026-11-13)
    _check_keys(ball, _BALL_KEYS | (_BALL_OPTIONAL & set(ball)), "s['ball']")
    passer = ball.get("passer")
    if passer is not None and (not isinstance(passer, int) or isinstance(passer, bool)
                               or not 0 <= passer < n_blue):
        raise ValueError(
            f"s['ball']['passer'] must be None or a blue id in [0, {n_blue}), got {passer!r}"
        )
    ### END AI GENERATED
    return StateArrays(
        blue_pos=blue[0],
        blue_vel=blue[1],
        blue_poss=blue[2],
        red_pos=red[0],
        red_vel=red[1],
        red_poss=red[2],
        ball_pos=_vec2(ball["pos"], "s['ball']['pos']"),
        ball_vel=_vec2(ball["vel"], "s['ball']['vel']"),
        passer=passer,  # AI GENERATED (2026-11-13)
        blue_calling=blue[3],  # AI GENERATED (2026-11-13)
    )


### AI GENERATED
def state_blocks(
    n_blue: int, n_red: int, passer_obs: bool = False, call_obs: bool = False
) -> list[tuple[str, int, int]]:
    """The flattening spec above as (block, numbers per slot, slots), in order. With
    ``passer_obs`` / ``call_obs`` (2026-11-13) each blue's slot ends with those flags."""
    return [("blues", 5 + int(passer_obs) + int(call_obs), n_blue), ("reds", 5, n_red),
            ("ball", 4, 1)]


def blue_flags(st: StateArrays, n_blue: int, passer_obs: bool, call_obs: bool) -> np.ndarray:
    """[N, k] float, in this order: 1 for the blue who passed the ball to its holder (if
    ``passer_obs``), 1 for each blue calling (if ``call_obs``). k = 0 with neither (2026-11-13)."""
    cols = []
    if passer_obs:
        passed = np.zeros(n_blue)
        if st.passer is not None:
            passed[st.passer] = 1.0
        cols.append(passed)
    if call_obs:
        cols.append(np.zeros(n_blue) if st.blue_calling is None
                    else st.blue_calling.astype(np.float64))
    return np.stack(cols, axis=1) if cols else np.zeros((n_blue, 0))
### END AI GENERATED


class StateFlattener:
    """Flattened, scaled global state for the critic (design #11)."""

    def __init__(self, cfg: SharedConfig, horizon_mode: str = "infinite", passer_obs: bool = False,
                 call_obs: bool = False):  # AI GENERATED (2026-11-13: passer_obs, call_obs)
        if horizon_mode != "infinite":
            raise NotImplementedError("horizon_mode='finite' (time remaining in s) is interface-only (#12)")
        self.n_blue, self.n_red = cfg.n_blue, cfg.n_red
        self._half = np.asarray(cfg.field_size, dtype=np.float64) / 2.0
        self._vmax = cfg.max_speed  # the single velocity unit
        self.passer_obs, self.call_obs = passer_obs, call_obs  # AI GENERATED (2026-11-13)
        # AI GENERATED
        self.dim = sum(width * slots for _, width, slots
                       in state_blocks(self.n_blue, self.n_red, passer_obs, call_obs))

    def __call__(self, st: StateArrays) -> np.ndarray:
        parts = []
        for team, (pos, vel, poss) in enumerate((
            (st.blue_pos, st.blue_vel, st.blue_poss),
            (st.red_pos, st.red_vel, st.red_poss),
        )):
            block = np.concatenate([pos / self._half, vel / self._vmax, poss[:, None].astype(np.float64)], axis=1)
            if team == 0:  # AI GENERATED (2026-11-13): the blues' flags last
                block = np.concatenate(
                    [block, blue_flags(st, self.n_blue, self.passer_obs, self.call_obs)], axis=1
                )
            parts.append(block.reshape(-1))  # row-major: entity by entity, in id order
        parts.append(st.ball_pos / self._half)
        parts.append(st.ball_vel / self._vmax)
        out = np.concatenate(parts).astype(np.float32)
        assert out.shape == (self.dim,)
        return out


def _check_keys(d: dict, expected: frozenset[str], where: str) -> None:
    keys = set(d)
    if keys != expected:
        unknown, missing = sorted(keys - expected), sorted(expected - keys)
        hint = " (new env state must be agreed and added to the spec, ICD §5)" if unknown else ""
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}{hint}")


def _entities(
    lst: Any, n: int, where: str, optional: frozenset[str] = frozenset()
) -> tuple[np.ndarray, ...]:
    if not isinstance(lst, list) or len(lst) != n:
        raise ValueError(f"{where} must be a list of length {n}")
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    poss = np.zeros(n, dtype=bool)
    calling = np.zeros(n, dtype=bool)  # AI GENERATED (2026-11-13)
    for i, e in enumerate(lst):
        if not isinstance(e, dict):
            raise TypeError(f"{where}[{i}] must be a dict")
        # AI GENERATED (optional keys)
        _check_keys(e, _ENTITY_KEYS | (optional & set(e)), f"{where}[{i}]")
        ### AI GENERATED (2026-11-13)
        if "calling" in e:
            if not isinstance(e["calling"], bool):
                raise TypeError(f"{where}[{i}]['calling'] must be a bool")
            calling[i] = e["calling"]
        ### END AI GENERATED
        if e["id"] != i:
            raise ValueError(f"{where}[{i}] has id {e['id']}; entities must be ordered by id (ICD §2)")
        if not isinstance(e["has_possession"], bool):
            raise TypeError(f"{where}[{i}]['has_possession'] must be a bool")
        pos[i] = _vec2(e["pos"], f"{where}[{i}]['pos']")
        vel[i] = _vec2(e["vel"], f"{where}[{i}]['vel']")
        poss[i] = e["has_possession"]
    return pos, vel, poss, calling  # AI GENERATED (2026-11-13: calling)


def _vec2(v: Any, where: str) -> np.ndarray:
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"{where} must be a list of 2 numbers")
    arr = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where} must be finite, got {v}")
    return arr
