# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: fixed-seed policy evaluation
#     with an isolated torch RNG (fork_rng), a generic episode runner for any
#     policy callable (reused by the baselines), and per-start-type summaries
#     (design #17, #23; walk-through steps 0 and 5).
#   - 2026-10-14: An episode's mean_reward now counts the base (task) reward
#     only, so shaping terms never change the evaluation (PBRS ablation). Added
#     an optional per-step observer to run_episodes and evaluate_actor, plus the
#     race metrics for the loose ball in free-ball starts (RaceTracker,
#     summarize_races: pickup rate, closing speed, alignment), moved from
#     Claude's pilot-analysis scripts at the student's request.
#   - 2026-11-07: actor_policy sends the kick heads' actions too. Each episode
#     gets its own reward memory (the passing drill's pass bookkeeping). Records
#     passes, kicks, return passes, pickups and calls per episode; summarize
#     adds passes_mean, frac_passed (episodes with at least one pass), pass_rate
#     (passes per kick, over all the episodes), return_share (returns per pass),
#     called_share and miscalled_per_pickup. Pylint cleanup: wrapped lines over
#     100 characters; no behaviour change. (Austin Beinder)
#   - 2026-11-25: Citation cleanup (backlog task 9): one Usage entry per PR,
#     oldest first; no behaviour change.
# ==============================================================================
"""Policy evaluation on fixed seeds (design #17, #23; walk-through steps 0 and 5).

* The SAME episodes every time: env seeds come from ``run.eval_seed`` only, never
  from the training seed, so every evaluation of every training run uses the
  same start states.
* A stochastic evaluation samples inside ``torch.random.fork_rng``, reseeded with
  ``eval_seed``. The training RNG stream is untouched, so turning evaluation on
  or off can't change training, and every evaluation point uses the same random
  numbers (common random numbers: lower-variance comparisons).
* Evaluation gets its OWN env instance: it must never reset the training env
  in the middle of a training episode.
* An optional per-step observer (``on_step``) sees every transition of the same
  episodes, e.g. ``RaceTracker``, which measures the race for a loose ball (the
  PBRS acquisition ablation) without playing any extra episodes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from marl.common.config import Config
from marl.common.spawn import START_FREE_BALL, START_POSSESSION, START_TARGET
from marl.rl.actions import actions_from_samples  # AI GENERATED (kick heads too)
from marl.rl.networks import Actor
from marl.rl.reward import RewardModule, effective_terminal
from marl.rl.rollout import EpisodeRecord, Featurizer, start_type_of

Policy = Callable[[dict], dict]  # global state s -> ICD action dict
StepObserver = Callable[[dict, dict, dict, bool, bool], None]  # (s, actions, s_next, terminated, truncated)
START_TYPES = (START_POSSESSION, START_FREE_BALL, START_TARGET)


def eval_env_seeds(eval_seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(np.random.SeedSequence(eval_seed))
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]


def actor_policy(actor: Actor, featurizer: Featurizer, deterministic: bool = False) -> Policy:
    def policy(s: dict) -> dict:
        obs, _ = featurizer(s)
        samples, _ = actor.act(torch.from_numpy(obs), deterministic=deterministic)
        return actions_from_samples(samples)  # AI GENERATED

    return policy


def run_episodes(
    env: Any,
    policy: Policy,
    cfg: Config,
    reward_fn: RewardModule,
    seeds: Sequence[int],
    on_step: StepObserver | None = None,
) -> list[EpisodeRecord]:
    """Play one full episode per seed with ``policy``; works for learned and scripted policies.
    ``on_step`` (if given) sees every transition; it can't change what is played."""
    options = cfg.shared.start.to_reset_options()
    records = []
    for seed in seeds:
        s, _ = env.reset(seed=int(seed), options=options)
        start_type = start_type_of(s, cfg.shared.scenario)
        base_sum = 0.0  # the task reward only: shaping is a training aid, never part of evaluation
        # AI GENERATED (2026-11-08): this episode's pass bookkeeping
        memory = reward_fn.new_episode()
        while True:
            actions = policy(s)
            s_next, terminated, truncated, info = env.step(actions)
            eff = effective_terminal(terminated, truncated, cfg.rl.horizon_mode)
            # AI GENERATED (memory)
            base_sum += reward_fn.parts(s, actions, s_next, eff, truncated, info, memory).base
            if on_step is not None:
                on_step(s, actions, s_next, terminated, truncated)
            s = s_next
            if terminated or truncated:
                break
        records.append(
            EpisodeRecord(
                length=s["step"],
                terminated=terminated,
                truncated=truncated,
                start_type=start_type,
                mean_reward=base_sum / s["step"],
                passes=memory.passes,  # AI GENERATED (2026-11-09)
                kicks=memory.kicks,  # AI GENERATED (2026-11-11)
                returns=memory.returns,  # AI GENERATED (2026-11-13)
                pickups=memory.pickups,  # AI GENERATED (2026-11-13)
                called=memory.called,  # AI GENERATED (2026-11-13)
                miscalled=memory.miscalled,  # AI GENERATED (2026-11-13)
            )
        )
    return records


def evaluate_actor(
    env: Any,
    cfg: Config,
    actor: Actor,
    featurizer: Featurizer,
    reward_fn: RewardModule,
    deterministic: bool = False,
    on_step: StepObserver | None = None,
) -> list[EpisodeRecord]:
    """The fixed evaluation protocol (#23): ``run.eval_episodes`` episodes on the eval seeds."""
    seeds = eval_env_seeds(cfg.run.eval_seed, cfg.run.eval_episodes)
    with torch.random.fork_rng(devices=[]):  # restore the training RNG afterwards
        torch.manual_seed(cfg.run.eval_seed)
        return run_episodes(env, actor_policy(actor, featurizer, deterministic), cfg, reward_fn, seeds, on_step)


def summarize(records: Sequence[EpisodeRecord]) -> dict[str, float]:
    """Episode statistics with a FIXED set of keys (CSV columns), NaN where undefined."""
    nan = float("nan")
    lengths = np.array([r.length for r in records], dtype=float)
    out: dict[str, float] = {
        "episodes": len(records),
        "len_mean": float(lengths.mean()) if len(records) else nan,
        "len_std": float(lengths.std()) if len(records) else nan,
        "frac_truncated": float(np.mean([r.truncated for r in records])) if len(records) else nan,
        "mean_reward": float(np.mean([r.mean_reward for r in records])) if len(records) else nan,
        ### AI GENERATED (2026-11-09)
        "passes_mean": float(np.mean([r.passes for r in records])) if len(records) else nan,
        "frac_passed": float(np.mean([r.passes > 0 for r in records])) if len(records) else nan,
        ### END AI GENERATED
        "pass_rate": _pass_rate(records),  # AI GENERATED (2026-11-11)
        "return_share": _return_share(records),  # AI GENERATED (2026-11-13)
        **_call_stats(records),  # AI GENERATED (2026-11-13)
    }
    for st in START_TYPES:
        sel = lengths[[r.start_type == st for r in records]] if len(records) else lengths
        out[f"n_{st}"] = len(sel)
        out[f"len_mean_{st}"] = float(sel.mean()) if len(sel) else nan
    return out


### AI GENERATED (2026-11-11)
def _pass_rate(records: Sequence[EpisodeRecord]) -> float:
    """Passes completed per kick over all the episodes (nan without kicks)."""
    kicks = sum(r.kicks for r in records)
    return sum(r.passes for r in records) / kicks if kicks else float("nan")


def _return_share(records: Sequence[EpisodeRecord]) -> float:
    """2026-11-13: the share of passes that went straight back to the blue who passed to the kicker
    (A -> B -> A; nan without passes). With three blues, 0 is a perfect rotation
    A -> B -> C -> A."""
    passes = sum(r.passes for r in records)
    return sum(r.returns for r in records) / passes if passes else float("nan")


def _call_stats(records: Sequence[EpisodeRecord]) -> dict[str, float]:
    """2026-11-13, per blue pickup of the ball: how well the blue who got it had called for it
    (called_share, 0 to 1) and how loudly the others had (miscalled_per_pickup). nan without
    pickups."""
    pickups = sum(r.pickups for r in records)
    if not pickups:
        return {"called_share": float("nan"), "miscalled_per_pickup": float("nan")}
    return {"called_share": sum(r.called for r in records) / pickups,
            "miscalled_per_pickup": sum(r.miscalled for r in records) / pickups}
### END AI GENERATED


# --- The race for a loose ball (PBRS acquisition ablation) ------------------------------
@dataclass(frozen=True)
class RaceRecord:
    """One free-ball start: did blue win the race for the loose ball, and how did it run?"""

    length: int  # episode length (control steps)
    pickup_step: int | None  # control step at which blue gained the ball; None = never
    approach_steps: int  # steps that began AND ended with the ball loose
    closed_m: float  # meters the closest blue closed on the ball over those steps
    moving_steps: int  # steps that began with the ball loose and that blue moved
    alignment_sum: float  # sum over those steps of cos(its motion, direction to the ball)


def _loose(s: dict) -> bool:
    return not any(p["has_possession"] for p in s["blue"]) and not any(p["has_possession"] for p in s["red"])


class RaceTracker:
    """A ``run_episodes`` observer: records the race for the loose ball in every free-ball
    start (``.records``). "Closest blue" is the team's closest player at each step, as in
    the PBRS potential. With no kicks (milestone 1) a loose ball doesn't move, so all of
    the distance closed is blue's doing."""

    def __init__(self, cfg: Config):
        self._scenario = cfg.shared.scenario
        self.records: list[RaceRecord] = []

    def __call__(self, s: dict, actions: dict, s_next: dict, terminated: bool, truncated: bool) -> None:
        if s["step"] == 0:  # an episode's first transition
            self._free = start_type_of(s, self._scenario) == START_FREE_BALL
            self._pickup: int | None = None
            self._n, self._closed, self._moving, self._cos = 0, 0.0, 0, 0.0
        if self._free and _loose(s):
            ball = s["ball"]["pos"]
            dists = [math.dist(b["pos"], ball) for b in s["blue"]]
            i = int(np.argmin(dists))
            if _loose(s_next):
                self._n += 1
                self._closed += dists[i] - min(math.dist(b["pos"], ball) for b in s_next["blue"])
            elif any(b["has_possession"] for b in s_next["blue"]):
                self._pickup = s_next["step"]
            p0, p1 = s["blue"][i]["pos"], s_next["blue"][i]["pos"]
            move = (p1[0] - p0[0], p1[1] - p0[1])
            norm = math.hypot(*move)
            if norm > 1e-9:
                self._moving += 1
                self._cos += (move[0] * (ball[0] - p0[0]) + move[1] * (ball[1] - p0[1])) / (norm * dists[i])
        if (terminated or truncated) and self._free:
            self.records.append(
                RaceRecord(s_next["step"], self._pickup, self._n, self._closed, self._moving, self._cos)
            )


def summarize_races(records: Sequence[RaceRecord]) -> dict[str, float]:
    """Race statistics with FIXED keys (CSV columns), NaN where undefined. Closing speed
    and alignment are pooled over steps, so a long race counts more than a short one."""
    nan = float("nan")
    won = [r.pickup_step for r in records if r.pickup_step is not None]
    steps, moving = sum(r.approach_steps for r in records), sum(r.moving_steps for r in records)
    return {
        "races": len(records),
        "pickup_rate": len(won) / len(records) if records else nan,
        "pickup_step_mean": float(np.mean(won)) if won else nan,
        "approach_m_per_step": sum(r.closed_m for r in records) / steps if steps else nan,
        "approach_alignment": sum(r.alignment_sum for r in records) / moving if moving else nan,
    }
