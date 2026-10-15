# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: evaluation tests (fixed seeds,
#     torch RNG isolation via fork_rng, reproducibility, a scripted policy with
#     an exactly known result, fixed summary columns).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
import math

import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.evaluate import eval_env_seeds, evaluate_actor, run_episodes, summarize
from marl.rl.networks import build_networks
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


def make(data=None, episodes=6):
    data = dict(data or {})
    data["run"] = {"eval_episodes": episodes}
    cfg = config_from_dict(data)
    torch.manual_seed(0)
    f = Featurizer(cfg)
    actor, _ = build_networks(cfg, f.obs_dim, f.state_dim)
    return cfg, f, actor, RewardModule(cfg)


def test_eval_seeds_are_fixed_plain_ints():
    a = eval_env_seeds(20261019, 5)
    assert a == eval_env_seeds(20261019, 5) and a != eval_env_seeds(1, 5)
    assert all(type(x) is int for x in a)


def test_evaluation_leaves_the_training_rng_untouched():
    cfg, f, actor, rf = make()
    state = torch.get_rng_state()
    evaluate_actor(MockEnv(cfg.shared), cfg, actor, f, rf)
    assert torch.equal(torch.get_rng_state(), state)


def test_evaluation_is_reproducible_and_modes_differ():
    cfg, f, actor, rf = make()
    env = MockEnv(cfg.shared)
    stoch1 = evaluate_actor(env, cfg, actor, f, rf)
    torch.rand(10)  # disturb the global RNG between calls: must not matter
    stoch2 = evaluate_actor(env, cfg, actor, f, rf)
    det = evaluate_actor(env, cfg, actor, f, rf, deterministic=True)
    assert stoch1 == stoch2
    assert det == evaluate_actor(env, cfg, actor, f, rf, deterministic=True)
    assert det != stoch1


def test_scripted_stand_still_in_goto_target_has_an_exact_result():
    cfg, f, actor, rf = make(GOTO)
    env = MockEnv(cfg.shared)
    still = lambda s: {"blue": [{"move": [0.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}]}
    seeds = eval_env_seeds(cfg.run.eval_seed, 3)
    records = run_episodes(env, still, cfg, rf, seeds)
    for seed, rec in zip(seeds, records):
        s0, _ = MockEnv(cfg.shared).reset(seed=seed, options=cfg.shared.start.to_reset_options())
        d0 = math.dist(s0["blue"][0]["pos"], s0["ball"]["pos"])
        assert rec.mean_reward == pytest.approx(-d0)  # never moves: distance stays d0 every step
        assert rec.length == cfg.shared.max_episode_steps and rec.truncated and rec.start_type == "target"


def test_summary_has_fixed_columns():
    cfg, f, actor, rf = make()
    records = evaluate_actor(MockEnv(cfg.shared), cfg, actor, f, rf)
    full, empty = summarize(records), summarize([])
    assert list(full) == list(empty)  # same CSV columns whatever happened
    assert full["episodes"] == 6 and empty["episodes"] == 0 and np.isnan(empty["len_mean"])
    assert full["n_possession"] + full["n_free_ball"] == 6 and full["n_target"] == 0
    assert np.isnan(full["len_mean_target"])
