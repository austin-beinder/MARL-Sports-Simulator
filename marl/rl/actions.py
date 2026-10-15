# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the action adapter from clamped
#     Beta samples to the ICD §6 action dict, and the commanded-speed metric
#     (design #18, #21, #23; KL-1 fix location).
# Prompt summary: Implement Phase 3 step 6 of the issue #7 plan (actor/critic
#   + Beta movement head + action adapter).
# Human review:   Pending review
# ==============================================================================
"""Action adapter: clamped Beta samples -> the ICD §6 action dict (plain Python).

    move fraction = 2x - 1   (x in [eps, 1 - eps] -> strictly inside [-1, 1])

Milestone 1 is movement-only (#21): ``kick_trigger = 0`` and
``kick_vec = [0, 0]`` always. The env scales the fractions and applies the
2-norm speed cap (ICD §6). If KL-1 (direction bias) ever needs fixing, the fix
goes in this module only (plan, "Known limitations").
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_env_actions(move01: Any) -> dict:
    """[N, 2] clamped movement samples in [0, 1] -> ``{"blue": [...]}`` (ICD §6)."""
    move01 = np.asarray(move01, dtype=np.float64)
    if move01.ndim != 2 or move01.shape[1] != 2:
        raise ValueError(f"expected movement samples of shape [N, 2], got {move01.shape}")
    if not np.all((move01 >= 0.0) & (move01 <= 1.0)):  # also rejects NaN
        raise ValueError("movement samples must lie in [0, 1] (clamped Beta samples)")
    frac = 2.0 * move01 - 1.0
    return {
        "blue": [
            {"move": [float(mx), float(my)], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}
            for mx, my in frac
        ]
    }


def commanded_speed_fraction(move01: Any) -> np.ndarray:
    """Speed the env will apply, as a fraction of the max (after its 2-norm cap).

    A logged metric (#23): persistently low values on axis-aligned moves are a
    KL-1 symptom.
    """
    frac = 2.0 * np.asarray(move01, dtype=np.float64) - 1.0
    return np.minimum(1.0, np.linalg.norm(frac, axis=-1))
