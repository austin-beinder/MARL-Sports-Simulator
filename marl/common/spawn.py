# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: the seeded, pure spawn sampler
#     (start type draw + rejection sampling under the design #17 / ICD §8-§9
#     constraints).
# Prompt summary: Implement Phase 3 step 2 of the issue #7 plan (spawn sampler)
#   as a small numpy-only module the env owner can reuse.
# Human review:   Pending review
# ==============================================================================
"""Reference spawn sampler (design #17, ICD §8-§9).

A pure function: (random generator, scenario + field parameters, reset options)
-> positions. numpy only (no torch), so the env owner can use it unchanged.

Start types:

* ``possession`` -- the ball starts on a blue player (held at step 0).
* ``free_ball``  -- the ball starts loose, closer to some blue than to any red
  (if ``require_blue_closer``), and outside every player's possession radius.
* ``target``     -- goto_target only: the ball is an immobile target.

Constraints (all start types): every player is inside the walls; any two players
are at least ``min_clearance`` apart; every red is at least
``min_red_ball_dist`` (> possession radius) from the ball, so no reset starts
with red possession or an unavoidable instant loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

START_POSSESSION = "possession"
START_FREE_BALL = "free_ball"
START_TARGET = "target"

RESET_OPTION_KEYS = frozenset({"start_mix", "require_blue_closer", "min_clearance", "min_red_ball_dist"})


@dataclass(frozen=True)
class Spawn:
    start_type: str
    blue_pos: tuple[tuple[float, float], ...]
    red_pos: tuple[tuple[float, float], ...]
    ball_pos: tuple[float, float]
    holder: int | None  # index of the blue holding the ball (possession starts only)


def sample_spawn(
    rng: np.random.Generator,
    *,
    scenario: str,
    n_blue: int,
    n_red: int,
    field_size: tuple[float, float],
    player_radius: float,
    possession_radius: float,
    options: dict[str, Any],
    max_tries: int = 10_000,
) -> Spawn:
    """Sample one start state. Deterministic for a given generator state.

    Raises ``ValueError`` for bad options and ``RuntimeError`` if the constraints
    can't be met (e.g. too many players for the field).
    """
    unknown = set(options) - RESET_OPTION_KEYS
    missing = RESET_OPTION_KEYS - set(options)
    if unknown or missing:
        raise ValueError(f"reset options: unknown {sorted(unknown)}, missing {sorted(missing)}")
    if scenario not in ("keepaway", "goto_target"):
        raise ValueError(f"unknown scenario {scenario!r}")

    clearance = float(options["min_clearance"])
    red_ball = float(options["min_red_ball_dist"])
    half = np.asarray(field_size, dtype=float) / 2.0
    lo, hi = -half + player_radius, half - player_radius

    # Draw the start type FIRST, so it depends only on the seed (ICD §2).
    if scenario == "goto_target":
        start_type, holder = START_TARGET, None
    elif rng.random() < float(options["start_mix"]):
        start_type, holder = START_POSSESSION, int(rng.integers(n_blue))
    else:
        start_type, holder = START_FREE_BALL, None

    for _ in range(max_tries):
        players = rng.uniform(lo, hi, size=(n_blue + n_red, 2))
        blue, red = players[:n_blue], players[n_blue:]
        ball = blue[holder].copy() if holder is not None else rng.uniform(lo, hi)
        if _ok(start_type, blue, red, ball, clearance, red_ball, possession_radius, options):
            return Spawn(
                start_type=start_type,
                blue_pos=tuple((float(x), float(y)) for x, y in blue),
                red_pos=tuple((float(x), float(y)) for x, y in red),
                ball_pos=(float(ball[0]), float(ball[1])),
                holder=holder,
            )
    raise RuntimeError(
        f"no valid {start_type} spawn after {max_tries} tries; the constraints are too tight "
        f"for field_size={tuple(field_size)} with {n_blue} blue + {n_red} red"
    )


def _ok(
    start_type: str,
    blue: np.ndarray,
    red: np.ndarray,
    ball: np.ndarray,
    clearance: float,
    red_ball: float,
    possession_radius: float,
    options: dict[str, Any],
) -> bool:
    players = np.concatenate([blue, red])
    if len(players) > 1:
        d = np.linalg.norm(players[:, None, :] - players[None, :, :], axis=-1)
        if d[np.triu_indices(len(players), k=1)].min() < clearance:
            return False
    d_red = np.linalg.norm(red - ball, axis=-1)
    if d_red.size and d_red.min() < red_ball:
        return False
    d_blue = np.linalg.norm(blue - ball, axis=-1)
    if start_type == START_FREE_BALL:
        if d_blue.min() <= possession_radius:  # genuinely loose, not an instant pickup
            return False
        if options["require_blue_closer"] and d_red.size and not d_blue.min() < d_red.min():
            return False
    elif start_type == START_TARGET:
        if d_blue.min() < clearance:  # don't start on top of the target
            return False
    return True
