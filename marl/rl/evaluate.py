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
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop); evaluation is called by the loop.
# Human review:   Pending review
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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from marl.common.config import Config
from marl.common.spawn import START_FREE_BALL, START_POSSESSION, START_TARGET
from marl.rl.actions import to_env_actions
from marl.rl.networks import MOVE, Actor
from marl.rl.reward import RewardModule, effective_terminal
from marl.rl.rollout import EpisodeRecord, Featurizer, start_type_of

Policy = Callable[[dict], dict]  # global state s -> ICD action dict
START_TYPES = (START_POSSESSION, START_FREE_BALL, START_TARGET)


def eval_env_seeds(eval_seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(np.random.SeedSequence(eval_seed))
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]


def actor_policy(actor: Actor, featurizer: Featurizer, deterministic: bool = False) -> Policy:
    def policy(s: dict) -> dict:
        obs, _ = featurizer(s)
        samples, _ = actor.act(torch.from_numpy(obs), deterministic=deterministic)
        return to_env_actions(samples[MOVE].numpy())

    return policy


def run_episodes(env: Any, policy: Policy, cfg: Config, reward_fn: RewardModule, seeds: Sequence[int]) -> list[EpisodeRecord]:
    """Play one full episode per seed with ``policy``; works for learned and scripted policies."""
    options = cfg.shared.start.to_reset_options()
    records = []
    for seed in seeds:
        s, _ = env.reset(seed=int(seed), options=options)
        start_type = start_type_of(s, cfg.shared.scenario)
        reward_sum = 0.0
        while True:
            actions = policy(s)
            s_next, terminated, truncated, info = env.step(actions)
            eff = effective_terminal(terminated, truncated, cfg.rl.horizon_mode)
            reward_sum += reward_fn(s, actions, s_next, eff, truncated, info)
            s = s_next
            if terminated or truncated:
                break
        records.append(
            EpisodeRecord(
                length=s["step"],
                terminated=terminated,
                truncated=truncated,
                start_type=start_type,
                mean_reward=reward_sum / reward_fn.scale / s["step"],
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
) -> list[EpisodeRecord]:
    """The fixed evaluation protocol (#23): ``run.eval_episodes`` episodes on the eval seeds."""
    seeds = eval_env_seeds(cfg.run.eval_seed, cfg.run.eval_episodes)
    with torch.random.fork_rng(devices=[]):  # restore the training RNG afterwards
        torch.manual_seed(cfg.run.eval_seed)
        return run_episodes(env, actor_policy(actor, featurizer, deterministic), cfg, reward_fn, seeds)


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
    }
    for st in START_TYPES:
        sel = lengths[[r.start_type == st for r in records]] if len(records) else lengths
        out[f"n_{st}"] = len(sel)
        out[f"len_mean_{st}"] = float(sel.mean()) if len(sel) else nan
    return out
