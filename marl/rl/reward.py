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
# Prompt summary: Implement Phase 3 step 5 of the issue #7 plan (reward
#   module + registry).
# Human review:   Pending review
# ==============================================================================
"""RL-side reward module (design #14, #15). The env never computes rewards.

    r = (1 - gamma) * (base + sum of enabled shaping terms * coef)

shared by all blue agents. Every term has the signature
``term(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float``,
fixed from day one so shaping terms that need any of these plug in unchanged.

Base terms:

* ``survival``    -- +1 every step, INCLUDING the step where red captures the
  ball. With the (1 - gamma) scale, an episode of L steps has return
  1 - gamma^L, in [0, 1) (#15).
* ``goto_target`` -- minus the blue-to-ball distance in meters, measured at
  ``s_next`` (the state the action led to).

Shaping terms: ``pbrs_red_distance`` (potential-based, #15) is interface-only
and raises ``NotImplementedError`` if enabled.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from marl.common.config import (
    BASE_REWARD_TERMS,
    SHAPING_TERMS_BUILT,
    SHAPING_TERMS_DEFERRED,
    Config,
    SharedConfig,
)

Term = Callable[[dict, dict, dict, bool, bool, dict, SharedConfig], float]


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


def _deferred(name: str, ref: str) -> Term:
    def term(*args: Any) -> float:
        raise NotImplementedError(f"reward term {name!r} is interface-only ({ref})")

    return term


BASE_TERMS: dict[str, Term] = {"survival": survival, "goto_target": goto_target}
SHAPING_TERMS: dict[str, Term] = {
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
        self._shared = cfg.shared

    def __call__(
        self,
        s: dict,
        actions: dict,
        s_next: dict,
        effective_terminal: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        args = (s, actions, s_next, effective_terminal, truncated, info, self._shared)
        total = self._base(*args)
        for term, coef in self._shaping:
            total += coef * term(*args)
        return self.scale * total


# The config validates names against its own constants; keep the two in sync.
assert set(BASE_TERMS) == set(BASE_REWARD_TERMS)
assert set(SHAPING_TERMS) == set(SHAPING_TERMS_BUILT | SHAPING_TERMS_DEFERRED)
