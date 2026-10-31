# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: evaluation tests (fixed seeds,
#     torch RNG isolation via fork_rng, reproducibility, a scripted policy with
#     an exactly known result, fixed summary columns).
#   - 2026-10-14: Added the race-metric tests (hand-built races with exact
#     answers, the observer never changes the episodes, scripted flee wins).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
import math

import numpy as np
import pytest
import torch

from marl.baselines import make_baseline
from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.evaluate import (
    RaceTracker,
    eval_env_seeds,
    evaluate_actor,
    run_episodes,
    summarize,
    summarize_races,
)
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


# --- The race for a loose ball (PBRS acquisition ablation) --------------------------------
def ka(step, blue, ball, red, holder=None):
    """A 1v1 state; ``holder`` is None (loose ball), ("blue", 0) or ("red", 0)."""

    def ent(team, pos):
        return {"id": 0, "pos": list(pos), "vel": [0.0, 0.0], "has_possession": holder == (team, 0)}

    return {"step": step, "blue": [ent("blue", blue)], "red": [ent("red", red)],
            "ball": {"pos": list(ball), "vel": [0.0, 0.0]}}


def test_race_tracker_measures_a_won_race_exactly():
    tr = RaceTracker(config_from_dict({}))
    s0 = ka(0, (2.0, 0.0), (0.0, 0.0), (8.0, 0.0))  # free-ball start, blue 2 m from the ball
    s1 = ka(1, (1.0, 0.0), (0.0, 0.0), (7.0, 0.0))  # straight at it: closes 1 m, cos 1
    s2 = ka(2, (1.0, 1.0), (0.0, 0.0), (6.0, 0.0))  # sideways: cos 0, "closes" 1 - sqrt(2)
    s3 = ka(3, (0.3, 0.0), (0.3, 0.0), (5.0, 0.0), holder=("blue", 0))  # the pickup
    s4 = ka(4, (0.4, 0.0), (0.4, 0.0), (0.5, 0.0), holder=("red", 0))  # red steals it later
    for a, b, term in ((s0, s1, False), (s1, s2, False), (s2, s3, False), (s3, s4, True)):
        tr(a, {}, b, term, False)
    (r,) = tr.records
    assert (r.length, r.pickup_step, r.approach_steps, r.moving_steps) == (4, 3, 2, 3)
    assert r.closed_m == pytest.approx(2 - math.sqrt(2))  # the pickup step isn't a pure approach
    assert r.alignment_sum == pytest.approx(1 + 0 + 1.7 / math.sqrt(1.49 * 2))


def test_race_tracker_lost_race_and_possession_start():
    tr = RaceTracker(config_from_dict({}))
    tr(ka(0, (5.0, 0.0), (0.0, 0.0), (-1.0, 0.0)), {},
       ka(1, (5.0, 0.0), (-0.45, 0.0), (-0.45, 0.0), holder=("red", 0)), True, False)  # red collects it
    tr(ka(0, (1.0, 1.0), (1.0, 1.0), (-6.0, 0.0), holder=("blue", 0)), {},
       ka(1, (1.1, 1.0), (1.1, 1.0), (-5.9, 0.0), holder=("blue", 0)), False, True)  # a possession start
    (r,) = tr.records  # only the free-ball start was a race
    assert (r.length, r.pickup_step, r.approach_steps, r.moving_steps) == (1, None, 0, 0)
    s = summarize_races(tr.records)
    assert s["races"] == 1 and s["pickup_rate"] == 0.0
    assert all(math.isnan(s[k]) for k in ("pickup_step_mean", "approach_m_per_step", "approach_alignment"))
    assert list(summarize_races([])) == list(s)  # fixed CSV columns


def test_race_tracker_observes_the_same_episodes_without_changing_them():
    cfg, f, actor, rf = make(episodes=20)
    env = MockEnv(cfg.shared)
    plain = evaluate_actor(env, cfg, actor, f, rf)
    tracker = RaceTracker(cfg)
    assert evaluate_actor(env, cfg, actor, f, rf, on_step=tracker) == plain
    assert [r.length for r in tracker.records] == [r.length for r in plain if r.start_type == "free_ball"]


def test_scripted_flee_runs_straight_at_a_loose_ball_and_always_wins():
    cfg = config_from_dict({"shared": {"start": {"start_mix": 0.0}}})
    tracker = RaceTracker(cfg)
    flee = make_baseline("flee", cfg.shared)
    run_episodes(MockEnv(cfg.shared), flee, cfg, RewardModule(cfg), eval_env_seeds(cfg.run.eval_seed, 20), tracker)
    s = summarize_races(tracker.records)
    assert s["races"] == 20 and s["pickup_rate"] == 1.0
    full_speed = cfg.shared.max_speed * cfg.shared.dt_control  # 0.1 m per control step
    assert s["approach_m_per_step"] == pytest.approx(full_speed) and s["approach_alignment"] == pytest.approx(1.0)
