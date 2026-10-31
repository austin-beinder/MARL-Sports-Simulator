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

Shaping terms are potential-based: F = gamma * Phi(s') - Phi(s) (Ng, Harada &
Russell 1999), with Phi(s') = 0 only when ``effective_terminal`` and the real
Phi(s') at a truncation or a rollout cut (#15). Then every policy's values shift
by exactly -Phi(s), so no advantage changes: shaping can speed up learning
(through the critic's errors) but never changes what is optimal.

* ``pbrs_ball_acquisition`` -- Phi = minus the distance (m) from the closest
  blue to a LOOSE ball; 0 while anyone holds it, so possession starts are
  unshaped. Built 2026-10-14 for the acquisition ablation.
* ``pbrs_red_distance`` -- the #15 potential (ball-to-nearest-red distance);
  interface-only, raises ``NotImplementedError`` if enabled.
"""

from __future__ import annotations

import math
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


class RewardParts(NamedTuple):
    """One transition's team reward, split for logging."""

    base: float  # unscaled base term: the task's own reward
    shaping: float  # unscaled sum of coef * term over the enabled shaping terms
    total: float  # what the learner receives: (1 - gamma) * (base + shaping)


def _deferred(name: str, ref: str) -> Term:
    def term(*args: Any) -> float:
        raise NotImplementedError(f"reward term {name!r} is interface-only ({ref})")

    return term


BASE_TERMS: dict[str, Term] = {"survival": survival, "goto_target": goto_target}
SHAPING_TERMS: dict[str, Term] = {
    "pbrs_ball_acquisition": pbrs_ball_acquisition,
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
        for t in rw.shaping:
            if t.name not in SHAPING_TERMS:
                raise ValueError(f"unknown shaping term {t.name!r}; choose from {sorted(SHAPING_TERMS)}")
            if not t.enabled:
                continue
            if t.name in SHAPING_TERMS_DEFERRED:
                raise NotImplementedError(f"shaping term {t.name!r} is interface-only (#15)")
            self._shaping.append((SHAPING_TERMS[t.name], t.coef))
        self.scale = 1.0 - cfg.gamma
        self._cfg = cfg

    def parts(
        self,
        s: dict,
        actions: dict,
        s_next: dict,
        effective_terminal: bool,
        truncated: bool,
        info: dict,
    ) -> RewardParts:
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
