# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the action adapter from clamped
#     Beta samples to the ICD §6 action dict, and the commanded-speed metric
#     (design #18, #21, #23; KL-1 fix location).
#   - 2026-11-07: Kick actions: to_env_actions takes the trigger and kick
#     samples, and actions_from_samples maps a whole sample dict. (Austin Beinder)
# Prompt summary: Implement Phase 3 step 6 of the issue #7 plan (actor/critic
#   + Beta movement head + action adapter).
# Human review:   Pending review
# ==============================================================================
"""Action adapter: clamped Beta samples -> the ICD §6 action dict (plain Python).

    move fraction = 2x - 1   (x in [eps, 1 - eps] -> strictly inside [-1, 1])
    kick_vec      = 2x - 1   (same map, from the kick_vec head)
    kick_trigger  = the Bernoulli sample as an int 0 / 1

Without kick heads (``kick_enabled = false``) every action has
``kick_trigger = 0`` and ``kick_vec = [0, 0]``. The env scales the fractions and
applies the 2-norm speed caps (ICD §6). If KL-1 (direction bias) ever needs
fixing, the fix goes in this module only (plan, "Known limitations").
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_env_actions(move01: Any, trigger: Any = None, kick01: Any = None) -> dict:
    """[N, 2] clamped movement samples in [0, 1] -> ``{"blue": [...]}`` (ICD §6).

    ``trigger`` ([N] or [N, 1], values 0 / 1) and ``kick01`` ([N, 2] clamped samples in
    [0, 1]) are the kick heads' samples; leave both out for "never kick".
    """
    move01 = _unit_samples(move01, "movement")
    frac = 2.0 * move01 - 1.0
    ### AI GENERATED (kick heads, 2026-11-07)
    n = len(frac)
    if (trigger is None) != (kick01 is None):
        raise ValueError("give both trigger and kick01, or neither")
    if trigger is None:
        triggers, kick = [0] * n, np.zeros((n, 2))
    else:
        t = np.asarray(trigger, dtype=np.float64).reshape(-1)
        if t.shape != (n,) or not np.all((t == 0.0) | (t == 1.0)):
            raise ValueError(f"expected {n} trigger samples, each 0 or 1, got {trigger!r}")
        triggers = [int(x) for x in t]
        kick = 2.0 * _unit_samples(kick01, "kick") - 1.0
        if len(kick) != n:
            raise ValueError(f"expected {n} kick samples, got {len(kick)}")
    return {
        "blue": [
            {"move": [float(mx), float(my)], "kick_trigger": trig, "kick_vec": [float(kx), float(ky)]}
            for (mx, my), trig, (kx, ky) in zip(frac, triggers, kick)
        ]
    }


def actions_from_samples(samples: dict[str, Any]) -> dict:
    """The actor's sample dict (``Actor.act``; tensors or arrays) -> the ICD action dict."""
    def arr(x: Any) -> np.ndarray:
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    if "kick_trigger" in samples:
        return to_env_actions(arr(samples["move"]), arr(samples["kick_trigger"]), arr(samples["kick_vec"]))
    return to_env_actions(arr(samples["move"]))


def _unit_samples(x: Any, what: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"expected {what} samples of shape [N, 2], got {x.shape}")
    if not np.all((x >= 0.0) & (x <= 1.0)):  # also rejects NaN
        raise ValueError(f"{what} samples must lie in [0, 1] (clamped Beta samples)")
    return x
### END AI GENERATED


def commanded_speed_fraction(move01: Any) -> np.ndarray:
    """Speed the env will apply, as a fraction of the max (after its 2-norm cap).

    A logged metric (#23): persistently low values on axis-aligned moves are a
    KL-1 symptom.
    """
    frac = 2.0 * np.asarray(move01, dtype=np.float64) - 1.0
    return np.minimum(1.0, np.linalg.norm(frac, axis=-1))
