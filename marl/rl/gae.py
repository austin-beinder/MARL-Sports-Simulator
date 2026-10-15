# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: two-mask GAE using a stored
#     next_value per step, vectorized over agent columns (design #12, #20).
# Prompt summary: Implement Phase 3 step 7 of the issue #7 plan (buffer +
#   GAE) with hand-built boundary tests.
# Human review:   Pending review
# ==============================================================================
"""Generalized Advantage Estimation with two masks (design #12, #20).

For each agent column i (vectorized over i):

    delta_t  = r_t + gamma * (1 - effective_terminal_t) * next_value_t - value_t
    A_t      = delta_t + gamma * lambda * (1 - episode_end_t) * A_{t+1},    A_T = 0
    return_t = A_t + value_t          (the critic's target: a lambda-return)

* ``next_value_t`` = V(true s_{t+1}), stored for EVERY step: the true final state
  at an episode end (captured before the reset), the live state at the rollout
  end. GAE never reads value_{t+1}, which after a reset belongs to the NEXT episode.
* Bootstrap mask (``effective_terminal``): drop gamma * V(s_{t+1}) only when the
  future value really is 0, i.e. red took the ball (or, in ``finite`` mode, a
  time-out). An ``infinite``-mode time-out still bootstraps: the game wasn't lost.
* Chain-cut mask (``episode_end`` = terminated OR truncated): stop the recursion
  at ANY episode end, because row t+1 starts a different episode. The rollout end
  is a chain cut too (A_T = 0); its bootstrap is carried by next_value_{T-1}.

Never merge the two masks into one ``done = terminated | truncated`` (CleanRL's
simplification): that also drops the bootstrap at every time-out, teaching the
critic that running out the clock is as bad as losing the ball.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    effective_terminal: np.ndarray,
    episode_end: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (advantages, returns), both float64 [T, N].

    ``rewards``: [T] (the shared team reward, broadcast to every agent) or [T, N].
    ``values``, ``next_values``: [T, N]. ``effective_terminal``, ``episode_end``: [T] bool.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [T, N], got {values.shape}")
    T, N = values.shape
    next_values = np.asarray(next_values, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    eff = np.asarray(effective_terminal, dtype=bool)
    end = np.asarray(episode_end, dtype=bool)
    if next_values.shape != (T, N) or rewards.shape not in ((T, 1), (T, N)):
        raise ValueError(f"shape mismatch: values {values.shape}, next_values {next_values.shape}, rewards {rewards.shape}")
    if eff.shape != (T,) or end.shape != (T,):
        raise ValueError(f"masks must have shape [{T}], got {eff.shape} and {end.shape}")
    if np.any(eff & ~end):
        raise ValueError("effective_terminal must imply episode_end (a real terminal also ends the episode)")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError(f"gamma and lambda must be in [0, 1], got {gamma}, {lam}")

    bootstrap = (~eff).astype(np.float64)[:, None]  # 0 where the future value is truly 0
    chain = (~end).astype(np.float64)  # 0 where row t+1 belongs to another episode
    deltas = rewards + gamma * bootstrap * next_values - values

    advantages = np.empty((T, N))
    running = np.zeros(N)  # A_T = 0: the rollout end is a chain cut
    for t in range(T - 1, -1, -1):
        running = deltas[t] + gamma * lam * chain[t] * running
        advantages[t] = running
    return advantages, advantages + values
