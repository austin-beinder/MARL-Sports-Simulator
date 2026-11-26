# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the RL-side actor observation
#     builder o_i = f(s, i) (self-first, mixed frame, per-type blocks with
#     possession + visible flags, distance ordering), design #13.
#   - 2026-11-07: Named the own-possession index (OWN_POSSESSION), which the
#     actor's kick masks read. obs_blocks, the layout as blocks, so a warm start
#     can grow the actor's first layer when the team sizes change.
#     rl.passer_obs: a "passed the ball to its holder" flag at the end of the
#     self block and of each teammate slot, and rl.call_enabled: a "calling"
#     flag after it. Pylint cleanup: wrapped lines over 100 characters; no
#     behaviour change. (Austin Beinder)
#   - 2026-11-25: Citation cleanup (backlog task 9): one Usage entry per PR,
#     oldest first; no behaviour change.
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

With ``passer_obs`` (2026-11-13) the self block and every teammate slot end with one more number:
1 if that blue passed the ball to its current (or latest) holder (``s["ball"]["passer"]``). So the
holder can tell its teammates apart: the one who just passed to it, and the one who didn't.
Dimension = 6 + 7 * (N - 1) + 6 * M + 6.

With ``call_obs`` (``rl.call_enabled``, 2026-11-13) they end with one more after that: 1 while that
blue is calling (``s["blue"][i]["calling"]``, its "got it" on the latest step).
"""

from __future__ import annotations

import numpy as np

from marl.common.config import SharedConfig
from marl.rl.state import StateArrays, blue_flags  # AI GENERATED (2026-11-13: blue_flags)

SELF_DIM = 5
SLOT_DIM = 6
# index of "I hold the ball" in o_i; the actor's kick masks read it  # AI GENERATED
OWN_POSSESSION = 4


### AI GENERATED
def obs_blocks(
    n_blue: int, n_red: int, passer_obs: bool = False, call_obs: bool = False
) -> list[tuple[str, int, int]]:
    """The layout above as (block, numbers per slot, slots), in order. ``passer_obs`` and
    ``call_obs`` (2026-11-13) add their flags at the end of the self block and of each teammate
    slot."""
    extra = int(passer_obs) + int(call_obs)
    return [("self", SELF_DIM + extra, 1), ("teammates", SLOT_DIM + extra, n_blue - 1),
            ("reds", SLOT_DIM, n_red), ("ball", SLOT_DIM, 1)]
### END AI GENERATED


class ObsBuilder:
    def __init__(self, cfg: SharedConfig, obs_ordering: str = "distance", obs_mode: str = "full",
                 # AI GENERATED (2026-11-13: passer_obs, call_obs)
                 passer_obs: bool = False, call_obs: bool = False):
        if obs_ordering != "distance":
            raise NotImplementedError(f"obs_ordering={obs_ordering!r} is interface-only (#13)")
        if obs_mode != "full":
            raise NotImplementedError(f"obs_mode={obs_mode!r} is interface-only (#13)")
        self.n_blue, self.n_red = cfg.n_blue, cfg.n_red
        self._half = np.asarray(cfg.field_size, dtype=np.float64) / 2.0
        self._vmax = cfg.max_speed
        self.passer_obs, self.call_obs = passer_obs, call_obs  # AI GENERATED (2026-11-13)
        # AI GENERATED
        self.dim = sum(width * slots for _, width, slots
                       in obs_blocks(self.n_blue, self.n_red, passer_obs, call_obs))

    def __call__(self, st: StateArrays) -> np.ndarray:
        """Observations for all blue agents at once: float32 array [N, dim]."""
        out = np.empty((self.n_blue, self.dim), dtype=np.float32)
        held = float(st.ball_held)
        ### AI GENERATED (2026-11-13: the passer and call flags, [N, k])
        flags = blue_flags(st, self.n_blue, self.passer_obs, self.call_obs)
        for i in range(self.n_blue):
            p, v = st.blue_pos[i], st.blue_vel[i]
            parts = [p / self._half, v / self._vmax, [float(st.blue_poss[i])], flags[i]]
            mates = np.array([j for j in range(self.n_blue) if j != i], dtype=int)
            parts.append(self._block(p, v, st.blue_pos[mates], st.blue_vel[mates],
                                     st.blue_poss[mates], flags[mates]))
            ### END AI GENERATED
            parts.append(self._block(p, v, st.red_pos, st.red_vel, st.red_poss))
            parts.append(self._slot(p, v, st.ball_pos, st.ball_vel, held))
            out[i] = np.concatenate(parts)
        return out

    def _block(self, p, v, pos, vel, poss, extra=None) -> np.ndarray:
        if len(pos) == 0:  # empty blocks are allowed (n_red = 0, or N = 1)
            return np.zeros(0)
        order = np.argsort(np.linalg.norm(pos - p, axis=1), kind="stable")  # nearest first; ties by id
        # AI GENERATED (extra)
        return np.concatenate([self._slot(p, v, pos[k], vel[k], float(poss[k]),
                                          () if extra is None else extra[k]) for k in order])

    def _slot(self, p, v, other_pos, other_vel, flag: float, extra=()) -> np.ndarray:
        visible = 1.0  # full observability
        # AI GENERATED (2026-11-13): extra = the slot's passer / call flags, if any, last
        return np.concatenate([(other_pos - p) / self._half, (other_vel - v) / self._vmax,
                               [flag, visible], extra])
