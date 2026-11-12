# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the RL-side reward module with a
#     term registry, (1 - gamma) scaling, the survival and goto_target base
#     terms, the interface-only PBRS term (design #14, #15), and the
#     effective_terminal helper (#12).
#   - 2026-10-14: Added the pbrs_ball_acquisition shaping term and its potential
#     (the student's acquisition-ablation spec), RewardModule.parts (base vs
#     shaping, for logging), and passed the full Config to the terms (PBRS needs
#     the learner's gamma).
#   - 2026-11-08: The passing drill: the passing_drill base term, the
#     pass_progress and pass_completed shaping terms, and PassMemory, the
#     per-episode pass bookkeeping they read. (Austin Beinder)
#   - 2026-11-08: passing_drill's hold reward grows with each catch: it pays
#     the number of passes completed so far, not a flat 1. (Austin Beinder)
#   - 2026-11-09: The pass_practice base term (one kick per episode: 1 for a
#     catch, else up to 0.5 for how close the ball came to a teammate), and
#     PassMemory's closest approach. (Austin Beinder)
#   - 2026-11-10: A minimum pass distance (MIN_PASS_DISTANCE, 1 m from the
#     kick point for a catch; the closest approach skips teammates nearer than
#     1.5 m), and pass_practice pays
#     a catch 0.75 to 1 by its length. (Austin Beinder)
#   - 2026-11-10: Both are options, off by default (rl.reward.min_pass_distance,
#     rl.reward.pass_length_bonus), so a stage can train on the old reward
#     first. (Austin Beinder)
#   - 2026-11-11: The kicker_clearance shaping term: the kicker pays for
#     staying near its own kick, and more for taking it back. (Austin Beinder)
# Prompt summary: Implement Phase 3 step 5 of the issue #7 plan (reward
#   module + registry).
# Human review:   Pending review
# ==============================================================================
"""RL-side reward module (design #14, #15). The env never computes rewards.

    r = (1 - gamma) * (base + sum of enabled shaping terms * coef)

shared by all blue agents. Every term has the signature
``term(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float``,
fixed from day one so shaping terms that need any of these plug in unchanged.
``cfg`` is the full Config (PBRS needs the learner's gamma).

Base terms:

* ``survival``    -- +1 every step, INCLUDING the step where red captures the
  ball. With the (1 - gamma) scale, an episode of L steps has return
  1 - gamma^L, in [0, 1) (#15).
* ``goto_target`` -- minus the blue-to-ball distance in meters, measured at
  ``s_next`` (the state the action led to).

The ``pbrs_*`` shaping terms are potential-based: F = gamma * Phi(s') - Phi(s) (Ng, Harada &
Russell 1999), with Phi(s') = 0 only when ``effective_terminal`` and the real
Phi(s') at a truncation or a rollout cut (#15). Then every policy's values shift
by exactly -Phi(s), so no advantage changes: shaping can speed up learning
(through the critic's errors) but never changes what is optimal.

* ``pbrs_ball_acquisition`` -- Phi = minus the distance (m) from the closest
  blue to a LOOSE ball; 0 while anyone holds it, so possession starts are
  unshaped. Built 2026-10-14 for the acquisition ablation.
* ``pbrs_red_distance`` -- the #15 potential (ball-to-nearest-red distance);
  interface-only, raises ``NotImplementedError`` if enabled.

The passing drill (2026-11-08; frozen blues, so passing is the only way to keep the ball):

* ``passing_drill`` (base) -- every step a blue holds the ball at ``s_next``, the number of
  passes the team has completed this episode: 1 per step after the first catch, 2 after the
  second, and so on, so every pass makes holding worth more. Holding the opening ball (no pass
  yet) earns nothing, and neither
  does a loose ball (kicking it away from a slow red would otherwise buy free time).
* ``pass_progress`` (shaping, NOT potential-based) -- while a blue's kick is in play, how much
  closer the ball got this step to the nearest OTHER blue (m). Kicking toward a teammate earns a
  little, away from them costs a little. It sums to (start distance - end distance) over a kick.
* ``pass_completed`` (shaping, NOT potential-based) -- 1 on the step a different blue picks up a
  blue's kick.

Pass practice (2026-11-09; one kick per episode, no red, frozen blues):

* ``pass_practice`` (base) -- paid once, on the episode's last step: 1 for a catch (with
  ``rl.reward.pass_length_bonus``, 2026-11-10: ``0.75 + 0.25 * min(1, length / 4 m)``, so longer
  passes pay a bit more), otherwise ``0.5 * exp(-gap / 1 m)``, where gap is how close the ball's path came to the
  edge of a teammate (closest approach, so a kick in the right direction scores even if it is
  too soft or too hard). 0 if the holder never kicked. A catch always beats a near miss.

A pass is a blue's kick picked up by a different blue with nobody else touching it first. With
``rl.reward.min_pass_distance`` d > 0 (2026-11-10), the catch must also be at least d from where
the ball was kicked, and the closest approach ignores teammates within d + 0.5 m of the kick
point, so a tap into a teammate next to the kicker earns nothing. That
needs memory across steps (who kicked the ball in play, how many passes so far), so these terms
read ``info["pass"]`` (a ``PassStep``), which ``RewardModule.parts`` adds from a ``PassMemory``.
Every loop that plays episodes owns one memory per episode (``RewardModule.new_episode``): the
training collector and the evaluator share a RewardModule and can be mid-episode at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass  # AI GENERATED
from typing import Any, Callable, NamedTuple

from marl.common.config import (
    BASE_REWARD_TERMS,
    SHAPING_TERMS_BUILT,
    SHAPING_TERMS_DEFERRED,
    Config,
)

Term = Callable[[dict, dict, dict, bool, bool, dict, Config], float]


def effective_terminal(terminated: bool, truncated: bool, horizon_mode: str) -> bool:
    """True when the future value is zero (#12): a real game over, or, in
    ``finite`` mode only, a time-out. Feeds the GAE bootstrap mask and PBRS."""
    return terminated or (horizon_mode == "finite" and truncated)


def survival(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return 1.0


def goto_target(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    ball = s_next["ball"]["pos"]
    dists = [math.dist(b["pos"], ball) for b in s_next["blue"]]
    return -sum(dists) / len(dists)


def ball_potential(s: dict) -> float:
    """Phi(s) for ``pbrs_ball_acquisition``: minus the distance (m) from the CLOSEST
    blue to the ball while it is loose (ICD §5: no ``has_possession`` set), 0 while
    anyone holds it. Closest only: a team potential, so at N > 1 it doesn't pull
    every teammate onto the ball."""
    if any(p["has_possession"] for p in s["blue"]) or any(p["has_possession"] for p in s["red"]):
        return 0.0
    ball = s["ball"]["pos"]
    return -min(math.dist(b["pos"], ball) for b in s["blue"])


def pbrs_ball_acquisition(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    """F = gamma * Phi(s') - Phi(s), with Phi(s') = 0 at an effective terminal (#15).

    Per unit coef, while the ball is loose and blue is d meters from it: a full-speed
    step toward it earns about gamma * max_speed * dt_control (0.099 at the defaults)
    on top of a drift of (1 - gamma) * d that every action gets (Phi < 0); a pickup
    earns about possession_radius; red capturing the ball refunds +d. These cancel
    over an episode: the discounted sum is gamma^L * Phi(s_L) - Phi(s_0) for EVERY
    policy (a test checks it)."""
    phi_next = 0.0 if effective_terminal else ball_potential(s_next)
    return cfg.gamma * phi_next - ball_potential(s)


### AI GENERATED (2026-11-08: the passing drill)
class PassStep(NamedTuple):
    """One step's pass bookkeeping, as the pass terms see it in ``info["pass"]``."""

    kicker: int | None  # the blue whose kick was in play during this step (None: no blue kick in play)
    completed: bool  # a different blue picked that kick up this step
    passes: int  # passes completed this episode, this step's included
    # 2026-11-09: the closest the ball's path has come to another blue's centre (m) since the
    # latest kick; inf before any kick
    closest: float = math.inf
    length: float = 0.0  # 2026-11-10: how far the latest completed pass went, kick point to catch point (m)


### AI GENERATED (2026-11-10: a minimum pass distance, rl.reward.min_pass_distance)
# With a minimum pass distance d > 0, a catch counts as a pass only if the ball is at least d
# from where it was kicked, and the closest approach ignores teammates within d + PASS_REACH of the
# kick point (they could only catch a shorter one; PASS_REACH: player 0.25 + ball 0.15 m, plus a
# margin). So a tap into a teammate next to the kicker counts for nothing (no pass, no near miss).
# d = 0 (the default) counts every catch and every teammate, as before.
PASS_REACH = 0.5
### END AI GENERATED


@dataclass
class PassMemory:
    """Per-episode memory for the pass terms; ``update`` it once per step, in order."""

    kicker: int | None = None  # the blue whose kick is in play (the ball is loose, or rebounded to them)
    passes: int = 0
    closest: float = math.inf  # 2026-11-09: see PassStep.closest
    kick_point: tuple[float, float] | None = None  # 2026-11-10: where the ball in play was kicked from
    length: float = 0.0  # 2026-11-10: see PassStep.length
    min_distance: float = 0.0  # 2026-11-10: rl.reward.min_pass_distance (m); 0 = no minimum

    def _far_enough(self, ball) -> bool:  # AI GENERATED (2026-11-10)
        """The ball is min_distance from the kick point (always, with no minimum or no kick point)."""
        return (self.min_distance <= 0 or self.kick_point is None
                or math.dist(ball, self.kick_point) >= self.min_distance)

    def _a_target(self, pos) -> bool:  # AI GENERATED (2026-11-10)
        """A teammate this far from the kick point could catch a pass of min_distance or more."""
        return (self.min_distance <= 0 or self.kick_point is None
                or math.dist(pos[:2], self.kick_point) >= self.min_distance + PASS_REACH)

    def update(self, info: dict, s: dict | None = None, s_next: dict | None = None) -> PassStep:
        """``s`` and ``s_next`` (2026-11-09) feed the closest approach; without them it stays inf.
        They also give the kick point (2026-11-10); without them every catch counts, however short."""
        in_play, completed = self.kicker, False
        for event in info["events"]:
            if event["type"] == "kick":
                self.kicker = in_play = event["agent"][1]  # blue kicks are the only kicks (ICD §6)
                self.closest = math.inf  # a new kick: measure its path from scratch
                self.kick_point = tuple(s["ball"]["pos"][:2]) if s is not None else None  # AI GENERATED (2026-11-10)
            elif event["type"] == "possession_change" and event["to"] is not None:
                to = event["to"]
                ### AI GENERATED (2026-11-10: a catch has to be min_distance from the kick point)
                caught_at = s_next["ball"]["pos"][:2] if s_next is not None else None
                if (to[0] == "blue" and self.kicker is not None and to[1] != self.kicker
                        and (caught_at is None or self._far_enough(caught_at))):
                    completed = True
                    self.passes += 1
                    self.length = math.dist(caught_at, self.kick_point) if caught_at and self.kick_point else 0.0
                ### END AI GENERATED
                self.kicker = None  # someone has the ball: no kick in play any more
        if in_play is not None and s is not None and s_next is not None:
            a, b = s["ball"]["pos"], s_next["ball"]["pos"]
            for blue in s_next["blue"]:
                if blue["id"] != in_play and self._a_target(blue["pos"]):  # AI GENERATED (2026-11-10)
                    self.closest = min(self.closest, _segment_distance(blue["pos"], a, b))
        return PassStep(in_play, completed, self.passes, self.closest, self.length)


def _segment_distance(p, a, b) -> float:
    """Distance from point p to the segment a-b (the ball's straight path over one step)."""
    ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


def _blue_holds(s: dict) -> bool:
    return any(b["has_possession"] for b in s["blue"])


def passing_drill(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return float(info["pass"].passes) if _blue_holds(s_next) else 0.0  # AI GENERATED (2026-11-08): grows per catch


def _teammate_distance(s: dict, kicker: int) -> float:
    """Distance (m) from the ball to the nearest blue other than the kicker."""
    ball = s["ball"]["pos"]
    return min(math.dist(b["pos"], ball) for b in s["blue"] if b["id"] != kicker)


def pass_progress(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    kicker = info["pass"].kicker
    if kicker is None or any(r["has_possession"] for r in s_next["red"]):
        return 0.0  # no kick in play, or red got it (the episode is over)
    return _teammate_distance(s, kicker) - _teammate_distance(s_next, kicker)


def pass_completed(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return 1.0 if info["pass"].completed else 0.0


### AI GENERATED (2026-11-11: the kicker keeps clear of its own kick)
# kicker_clearance: while a kick is in play, each step costs the kicker's closeness to the ball,
# 1 - distance / KICKER_CLEARANCE (0 from KICKER_CLEARANCE out, nearly 1 on top of it), and the
# kicker taking the ball back costs KICKER_REGRAB at once (it ends the episode, so the steps it
# saves would otherwise make a re-grab the cheap way out). So the kicker learns to kick firmly and
# leave the ball to its teammates instead of chasing it. A negative term with a positive coef,
# like every shaping coef; with coef c an episode spent on the ball costs about c per step.
# The kicker isn't marked in o_i: it has to tell from the ball running away from it.
KICKER_CLEARANCE = 1.5  # m, centre to centre
KICKER_REGRAB = 10.0  # a re-grab costs as much as 10 steps right on the ball


def kicker_clearance(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    kicker = info["pass"].kicker
    if kicker is None:
        return 0.0  # no kick in play
    me = next(b for b in s_next["blue"] if b["id"] == kicker)
    if me["has_possession"]:
        return -KICKER_REGRAB  # took its own kick back (or it rebounded straight to it)
    gap = math.dist(me["pos"][:2], s_next["ball"]["pos"][:2])
    return -max(0.0, 1.0 - gap / KICKER_CLEARANCE)
### END AI GENERATED


### AI GENERATED (2026-11-09: pass practice)
PASS_MISS_MAX = 0.5  # the most a missed kick can earn (a catch earns 1, or 0.75 to 1 by length)
PASS_MISS_SCALE = 1.0  # m: a miss's reward halves every ~0.7 m it passes wide of a teammate
### AI GENERATED (2026-11-10: longer catches pay a bit more, rl.reward.pass_length_bonus)
# With the bonus on, a catch pays PASS_CATCH_MIN + (1 - PASS_CATCH_MIN) * min(1, length /
# PASS_FULL_LENGTH): 0.81 for a 1 m pass, 1 from 4 m on. Off, every catch pays 1. Either way a
# catch beats any miss (PASS_MISS_MAX).
PASS_CATCH_MIN = 0.75
PASS_FULL_LENGTH = 4.0  # m
assert PASS_CATCH_MIN > PASS_MISS_MAX
### END AI GENERATED


def pass_practice(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    if not (effective_terminal or truncated):
        return 0.0  # paid once, when the kick has played out (or time ran out)
    p = info["pass"]
    if p.passes:
        ### AI GENERATED (2026-11-10: the length bonus)
        if not cfg.rl.reward.pass_length_bonus:
            return 1.0
        return PASS_CATCH_MIN + (1.0 - PASS_CATCH_MIN) * min(1.0, p.length / PASS_FULL_LENGTH)
        ### END AI GENERATED
    if math.isinf(p.closest):
        return 0.0  # never kicked
    gap = max(0.0, p.closest - cfg.shared.player_radius)
    return PASS_MISS_MAX * math.exp(-gap / PASS_MISS_SCALE)
### END AI GENERATED


PASS_TERMS = frozenset({"passing_drill", "pass_progress", "pass_completed", "pass_practice",  # the terms that read info["pass"]
                        "kicker_clearance"})  # AI GENERATED (2026-11-11)
### END AI GENERATED


class RewardParts(NamedTuple):
    """One transition's team reward, split for logging."""

    base: float  # unscaled base term: the task's own reward
    shaping: float  # unscaled sum of coef * term over the enabled shaping terms
    total: float  # what the learner receives: (1 - gamma) * (base + shaping)


def _deferred(name: str, ref: str) -> Term:
    def term(*args: Any) -> float:
        raise NotImplementedError(f"reward term {name!r} is interface-only ({ref})")

    return term


BASE_TERMS: dict[str, Term] = {
    "survival": survival,
    "goto_target": goto_target,
    "passing_drill": passing_drill,  # AI GENERATED (2026-11-08)
    "pass_practice": pass_practice,  # AI GENERATED (2026-11-09)
}
SHAPING_TERMS: dict[str, Term] = {
    "pbrs_ball_acquisition": pbrs_ball_acquisition,
    "pass_progress": pass_progress,  # AI GENERATED (2026-11-08)
    "pass_completed": pass_completed,  # AI GENERATED (2026-11-08)
    "kicker_clearance": kicker_clearance,  # AI GENERATED (2026-11-11)
    "pbrs_red_distance": _deferred("pbrs_red_distance", "#15"),
}


class RewardModule:
    """Computes the scaled team reward for one transition."""

    def __init__(self, cfg: Config):
        rw = cfg.rl.reward
        if rw.base not in BASE_TERMS:
            raise ValueError(f"unknown base reward term {rw.base!r}; choose from {sorted(BASE_TERMS)}")
        self._base = BASE_TERMS[rw.base]
        self._shaping: list[tuple[Term, float]] = []
        self.needs_memory = rw.base in PASS_TERMS  # AI GENERATED (2026-11-08)
        for t in rw.shaping:
            if t.name not in SHAPING_TERMS:
                raise ValueError(f"unknown shaping term {t.name!r}; choose from {sorted(SHAPING_TERMS)}")
            if not t.enabled:
                continue
            if t.name in SHAPING_TERMS_DEFERRED:
                raise NotImplementedError(f"shaping term {t.name!r} is interface-only (#15)")
            self._shaping.append((SHAPING_TERMS[t.name], t.coef))
            self.needs_memory |= t.name in PASS_TERMS  # AI GENERATED (2026-11-08)
        self.scale = 1.0 - cfg.gamma
        self._cfg = cfg

    ### AI GENERATED (2026-11-08)
    def new_episode(self) -> PassMemory:
        """Fresh per-episode memory, to pass to every ``parts`` call of that episode."""
        return PassMemory(min_distance=self._cfg.rl.reward.min_pass_distance)  # 2026-11-10
    ### END AI GENERATED

    def parts(
        self,
        s: dict,
        actions: dict,
        s_next: dict,
        effective_terminal: bool,
        truncated: bool,
        info: dict,
        memory: PassMemory | None = None,  # AI GENERATED (2026-11-08)
    ) -> RewardParts:
        ### AI GENERATED (2026-11-08: the pass terms read info["pass"], built from the episode's memory)
        if memory is not None:
            info = {**info, "pass": memory.update(info, s, s_next)}  # s, s_next: closest approach (2026-11-09)
        elif self.needs_memory:
            raise ValueError("this reward has pass terms: pass the episode's memory (RewardModule.new_episode())")
        ### END AI GENERATED
        args = (s, actions, s_next, effective_terminal, truncated, info, self._cfg)
        base = self._base(*args)
        shaping = 0.0
        for term, coef in self._shaping:
            shaping += coef * term(*args)
        # Without shaping, base + 0.0 == base exactly: survival-only runs reproduce bit for bit.
        return RewardParts(base, shaping, self.scale * (base + shaping))

    def __call__(
        self,
        s: dict,
        actions: dict,
        s_next: dict,
        effective_terminal: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        return self.parts(s, actions, s_next, effective_terminal, truncated, info).total


# The config validates names against its own constants; keep the two in sync.
assert set(BASE_TERMS) == set(BASE_REWARD_TERMS)
assert set(SHAPING_TERMS) == set(SHAPING_TERMS_BUILT | SHAPING_TERMS_DEFERRED)
