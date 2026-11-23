# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: the seeded, pure spawn sampler
#     (start type draw + rejection sampling under the design #17 / ICD §8-§9
#     constraints).
#   - 2026-11-07: The max_ball_dist reset option: a free ball spawns within
#     that distance of a blue player (0 = anywhere). The curriculum grows it.
#     (Austin Beinder)
#   - 2026-11-08: The passing_drill spawn: blues on a circle around red, one
#     of them holding the ball, and the drill_radius option. (Austin Beinder)
#   - 2026-11-09: The pass_practice spawn (teammates pass_distance from the
#     holder, no red) and the pass_distance option. (Austin Beinder)
#   - 2026-11-09: pass_practice_moving uses the same spawn. (Austin Beinder)
#   - 2026-11-11: The catch_start_mix option: a share of keepaway resets use
#     the pass practice layout, plus red. (Austin Beinder)
#   - 2026-11-14: claim_ball: always a loose ball. (Austin Beinder)
#   - 2026-11-23: soccer: drawn exactly like keepaway. (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters; no
#     behaviour change. (Austin Beinder)
# ==============================================================================
"""Reference spawn sampler (design #17, ICD §8-§9).

A pure function: (random generator, scenario + field parameters, reset options)
-> positions. numpy only (no torch), so the env owner can use it unchanged.

Start types:

* ``possession`` -- the ball starts on a blue player (held at step 0).
* ``free_ball``  -- the ball starts loose, closer to some blue than to any red
  (if ``require_blue_closer``), and outside every player's possession radius.
* ``target``     -- goto_target only: the ball is an immobile target.

passing_drill (2026-11-08) always starts with a blue holding the ball (``possession``), and its
own layout: the blues spread evenly round a circle of about ``drill_radius`` metres (radius
+-15%, each angle +-15 deg) with red near the middle (within 30% of the radius). Only
``min_clearance`` and ``drill_radius`` apply; ``start_mix``, ``require_blue_closer``,
``min_red_ball_dist`` and ``max_ball_dist`` don't (red always starts about one radius from the
carrier).

pass_practice and pass_practice_moving (2026-11-09) also always start with a blue holding the
ball, with no red. Every other blue stands ``pass_distance`` metres (+-15%) from the holder, in a
random direction. Only ``min_clearance`` and ``pass_distance`` apply.

``catch_start_mix`` (keepaway only, 2026-11-11): the share of resets that start like pass
practice -- a random blue holding the ball, every other blue ``pass_distance`` (+-15%) from it --
with each red anywhere at least ``min_red_ball_dist`` from the ball (and ``min_clearance`` from
everyone). The other resets are the usual keepaway starts (``start_mix``, ``max_ball_dist``, ...).
A curriculum steps it from 1 down to 0, so a policy trained on pass practice meets red in the
layout it knows first. With 0 the draws are exactly the same as before the option existed.

Constraints (all start types): every player is inside the walls; any two players
are at least ``min_clearance`` apart; every red is at least
``min_red_ball_dist`` (> possession radius) from the ball, so no reset starts
with red possession or an unavoidable instant loss.

``max_ball_dist`` (free-ball starts only, 2026-11-07): if > 0, the ball spawns
uniformly in a disc of that radius around a random blue player, instead of
anywhere on the field. Small values make the start easy (the ball is right
there); 0 means anywhere, the real task. The curriculum
(``curriculum/runner.py``) grows it, like the goal radius in CACTUS (Phan et
al. 2024). With 0, the draws are exactly the same as before the option existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

START_POSSESSION = "possession"
START_FREE_BALL = "free_ball"
START_TARGET = "target"

RESET_OPTION_KEYS = frozenset(
    {"start_mix", "require_blue_closer", "min_clearance", "min_red_ball_dist",
     # AI GENERATED (max_ball_dist)
     "max_ball_dist",
     "drill_radius",  # AI GENERATED (2026-11-08)
     "pass_distance",  # AI GENERATED (2026-11-09)
     "catch_start_mix"}  # AI GENERATED (2026-11-11)
)


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
    # AI GENERATED (drills)
    if scenario not in ("keepaway", "goto_target", "passing_drill", "pass_practice",
                        "pass_practice_moving",
                        "claim_ball",  # AI GENERATED (2026-11-14)
                        "soccer"):  # AI GENERATED (2026-11-23): drawn exactly like keepaway
        raise ValueError(f"unknown scenario {scenario!r}")

    clearance = float(options["min_clearance"])
    red_ball = float(options["min_red_ball_dist"])
    ### AI GENERATED
    max_ball = float(options["max_ball_dist"])  # 0 = anywhere
    if max_ball < 0:
        raise ValueError(f"max_ball_dist must be >= 0 (0 = anywhere), got {max_ball}")
    ### END AI GENERATED
    half = np.asarray(field_size, dtype=float) / 2.0
    lo, hi = -half + player_radius, half - player_radius
    ### AI GENERATED (2026-11-08)
    if scenario == "passing_drill":
        return _drill_spawn(rng, n_blue, n_red, lo, hi, clearance,
                            float(options["drill_radius"]), max_tries)
    if scenario in ("pass_practice", "pass_practice_moving"):  # 2026-11-09
        return _practice_spawn(rng, n_blue, n_red, lo, hi, clearance,
                               float(options["pass_distance"]), max_tries)
    ### END AI GENERATED
    ### AI GENERATED (2026-11-11: catch_start_mix)
    catch_mix = float(options["catch_start_mix"])
    if not 0.0 <= catch_mix <= 1.0:
        raise ValueError(f"catch_start_mix must be in [0, 1], got {catch_mix}")
    # Drawn only when the option is on, so catch_start_mix = 0 leaves every other draw unchanged
    if scenario == "keepaway" and catch_mix > 0 and rng.random() < catch_mix:
        return _practice_spawn(rng, n_blue, n_red, lo, hi, clearance,
                               float(options["pass_distance"]), max_tries,
                               red_ball=red_ball)
    ### END AI GENERATED

    # Draw the start type FIRST, so it depends only on the seed (ICD §2).
    if scenario == "goto_target":
        start_type, holder = START_TARGET, None
    # AI GENERATED (2026-11-14): always a loose ball (start_mix doesn't apply)
    elif scenario == "claim_ball":
        start_type, holder = START_FREE_BALL, None
    elif rng.random() < float(options["start_mix"]):
        start_type, holder = START_POSSESSION, int(rng.integers(n_blue))
    else:
        start_type, holder = START_FREE_BALL, None

    for _ in range(max_tries):
        players = rng.uniform(lo, hi, size=(n_blue + n_red, 2))
        blue, red = players[:n_blue], players[n_blue:]
        ### AI GENERATED (the max_ball_dist branch)
        if holder is not None:
            ball = blue[holder].copy()
        elif start_type == START_FREE_BALL and max_ball > 0:
            ball = _ball_near(rng, blue, max_ball)
            if np.any(ball < lo) or np.any(ball > hi):
                continue  # outside the walls
        else:
            ball = rng.uniform(lo, hi)
        ### END AI GENERATED
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


### AI GENERATED
DRILL_RADIUS_JITTER = 0.15  # the circle's radius varies +-15%
DRILL_ANGLE_JITTER = np.radians(15.0)  # each blue's angle varies +-15 deg from even spacing
DRILL_RED_SPREAD = 0.3  # red starts within 30% of the radius from the circle's centre


def _drill_spawn(rng, n_blue, n_red, lo, hi, clearance, radius, max_tries) -> Spawn:
    """passing_drill: blues round a circle, red near its centre, a random blue holding the ball."""
    if radius <= 0:
        raise ValueError(f"drill_radius must be > 0, got {radius}")
    holder = int(rng.integers(n_blue))  # drawn first, like the start type in sample_spawn
    for _ in range(max_tries):
        r = radius * rng.uniform(1 - DRILL_RADIUS_JITTER, 1 + DRILL_RADIUS_JITTER)
        angles = (rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(n_blue) / n_blue
                  + rng.uniform(-DRILL_ANGLE_JITTER, DRILL_ANGLE_JITTER, size=n_blue))
        center = rng.uniform(lo, hi)
        blue = center + r * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        spread = DRILL_RED_SPREAD * r * np.sqrt(rng.random(n_red))
        red_angles = rng.uniform(0, 2 * np.pi, size=n_red)
        red = center + spread[:, None] * np.stack([np.cos(red_angles), np.sin(red_angles)], axis=-1)
        players = np.concatenate([blue, red])
        if np.any(players < lo) or np.any(players > hi):
            continue  # outside the walls
        d = np.linalg.norm(players[:, None, :] - players[None, :, :], axis=-1)
        if d[np.triu_indices(len(players), k=1)].min() < clearance:
            continue
        ball = blue[holder]
        return Spawn(
            start_type=START_POSSESSION,
            blue_pos=tuple((float(x), float(y)) for x, y in blue),
            red_pos=tuple((float(x), float(y)) for x, y in red),
            ball_pos=(float(ball[0]), float(ball[1])),
            holder=holder,
        )
    raise RuntimeError(
        f"no valid passing_drill spawn after {max_tries} tries; drill_radius={radius} doesn't fit "
        f"{n_blue} blue + {n_red} red at min_clearance={clearance}"
    )


PASS_DISTANCE_JITTER = 0.15  # 2026-11-09: each teammate stands pass_distance +-15% from the holder


def _practice_spawn(rng, n_blue, n_red, lo, hi, clearance, distance, max_tries,
                    red_ball=None) -> Spawn:
    """pass_practice (2026-11-09): a random blue holds the ball; every other blue stands about
    ``distance`` m from it (+-15%) in a random direction. No red.

    With ``red_ball`` (2026-11-11, keepaway's catch_start_mix) the reds are placed too: anywhere
    at least ``red_ball`` from the ball and ``clearance`` from every player."""
    if distance <= 0:
        raise ValueError(f"pass_distance must be > 0, got {distance}")
    if n_red and red_ball is None:
        raise ValueError(f"pass_practice has no red players, got n_red={n_red}")
    holder = int(rng.integers(n_blue))  # drawn first, like the start type in sample_spawn
    for _ in range(max_tries):
        origin = rng.uniform(lo, hi)
        n_other = n_blue - 1
        dist = distance * rng.uniform(1 - PASS_DISTANCE_JITTER, 1 + PASS_DISTANCE_JITTER,
                                      size=n_other)
        angles = rng.uniform(0, 2 * np.pi, size=n_other)
        others = origin + dist[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        blue = np.insert(others, holder, origin, axis=0)
        if np.any(blue < lo) or np.any(blue > hi):
            continue  # outside the walls
        ### AI GENERATED (2026-11-11: reds for catch_start_mix; none drawn without them)
        red = rng.uniform(lo, hi, size=(n_red, 2)) if n_red else np.zeros((0, 2))
        if n_red and np.linalg.norm(red - origin, axis=-1).min() < red_ball:
            continue  # red starts too close to the ball
        players = np.concatenate([blue, red])
        d = np.linalg.norm(players[:, None, :] - players[None, :, :], axis=-1)
        if len(players) > 1 and d[np.triu_indices(len(players), k=1)].min() < clearance:
            continue
        ### END AI GENERATED
        return Spawn(
            start_type=START_POSSESSION,
            blue_pos=tuple((float(x), float(y)) for x, y in blue),
            red_pos=tuple((float(x), float(y)) for x, y in red),  # AI GENERATED (2026-11-11)
            ball_pos=(float(origin[0]), float(origin[1])),
            holder=holder,
        )
    raise RuntimeError(
        f"no valid pass_practice spawn after {max_tries} tries; "
        f"pass_distance={distance} doesn't fit "
        f"{n_blue} blue + {n_red} red at min_clearance={clearance}"
    )


def _ball_near(rng: np.random.Generator, blue: np.ndarray, radius: float) -> np.ndarray:
    """A point uniform in the disc of this radius around a random blue player."""
    center = blue[int(rng.integers(len(blue)))]
    # sqrt: uniform over the disc's area, not bunched at the centre
    r = radius * np.sqrt(rng.random())
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return center + r * np.array([np.cos(angle), np.sin(angle)])
### END AI GENERATED


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
