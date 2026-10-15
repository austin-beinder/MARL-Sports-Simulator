# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the RL-side actor observation
#     builder o_i = f(s, i) (self-first, mixed frame, per-type blocks with
#     possession + visible flags, distance ordering), design #13.
# Prompt summary: Implement Phase 3 step 4 of the issue #7 plan (state
#   flattener + obs builder).
# Human review:   Pending review
# ==============================================================================
"""Actor observations o_i = f(s, i), built RL-side (design #13).

Layout for blue agent i (mixed frame, per-type blocks):

    self (5):                abs pos / half_field (2), vel / max_speed (2), own has_possession (1)
    teammates (N-1 slots x 6) \\
    reds      (M slots x 6)    > per slot: rel pos / half_field (2), rel vel / max_speed (2),
    ball      (1 slot x 6)    /            possession flag (1), visible flag (1)

* "rel" = other minus self. Relative velocities all use ``max_speed`` as the scale.
* Ball slot possession flag = "held by anyone" (a loose ball = 0).
* ``visible`` is always 1 under ``obs_mode = full``. ``radius`` mode would only
  change these flags and zero the hidden slots, so the size never changes.
* Within the teammate and red blocks, slots are sorted by distance to agent i
  (nearest first; ties by id) -- ``obs_ordering = distance``.
* Self-first REPLACES an agent ID: the shared actor knows "which one am I"
  because slot 0 is always "me" (#4).

Dimension = 5 + 6 * (N - 1) + 6 * M + 6.
"""

from __future__ import annotations

import numpy as np

from marl.common.config import SharedConfig
from marl.rl.state import StateArrays

SELF_DIM = 5
SLOT_DIM = 6


class ObsBuilder:
    def __init__(self, cfg: SharedConfig, obs_ordering: str = "distance", obs_mode: str = "full"):
        if obs_ordering != "distance":
            raise NotImplementedError(f"obs_ordering={obs_ordering!r} is interface-only (#13)")
        if obs_mode != "full":
            raise NotImplementedError(f"obs_mode={obs_mode!r} is interface-only (#13)")
        self.n_blue, self.n_red = cfg.n_blue, cfg.n_red
        self._half = np.asarray(cfg.field_size, dtype=np.float64) / 2.0
        self._vmax = cfg.max_speed
        self.dim = SELF_DIM + SLOT_DIM * (self.n_blue - 1) + SLOT_DIM * self.n_red + SLOT_DIM

    def __call__(self, st: StateArrays) -> np.ndarray:
        """Observations for all blue agents at once: float32 array [N, dim]."""
        out = np.empty((self.n_blue, self.dim), dtype=np.float32)
        held = float(st.ball_held)
        for i in range(self.n_blue):
            p, v = st.blue_pos[i], st.blue_vel[i]
            parts = [p / self._half, v / self._vmax, [float(st.blue_poss[i])]]
            mates = np.array([j for j in range(self.n_blue) if j != i], dtype=int)
            parts.append(self._block(p, v, st.blue_pos[mates], st.blue_vel[mates], st.blue_poss[mates]))
            parts.append(self._block(p, v, st.red_pos, st.red_vel, st.red_poss))
            parts.append(self._slot(p, v, st.ball_pos, st.ball_vel, held))
            out[i] = np.concatenate(parts)
        return out

    def _block(self, p, v, pos, vel, poss) -> np.ndarray:
        if len(pos) == 0:  # empty blocks are allowed (n_red = 0, or N = 1)
            return np.zeros(0)
        order = np.argsort(np.linalg.norm(pos - p, axis=1), kind="stable")  # nearest first; ties by id
        return np.concatenate([self._slot(p, v, pos[k], vel[k], float(poss[k])) for k in order])

    def _slot(self, p, v, other_pos, other_vel, flag: float) -> np.ndarray:
        visible = 1.0  # full observability
        return np.concatenate([(other_pos - p) / self._half, (other_vel - v) / self._vmax, [flag, visible]])
