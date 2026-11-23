# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-23
# Author:  Austin Beinder
# Usage:
#   - 2026-11-23: Generated this entire file: tests for the soccer scenario
#     (docs/soccer_env_plan.md §7): config validation, make_env, the adapter's
#     goal events and episode ends, the goal reward and shaping, the goals
#     observation block, warm starts from keep-away, outcome metrics, the
#     dribble baseline and a tiny training run.
# ==============================================================================
### AI GENERATED
"""The soccer scenario, RL side (the game itself: pybulletGame/tests/soccer_env_test.py)."""

import csv
import math

import numpy as np
import pytest
import torch

from marl.baselines import baselines_for, make_baseline
from marl.common.config import config_from_dict, config_to_dict
from marl.envs import make_env
from marl.envs.pybullet_soccer_env import PyBulletSoccerEnv
from marl.rl.checkpoint import warm_start
from marl.rl.evaluate import run_episodes, summarize
from marl.rl.networks import MOVE, build_networks
from marl.rl.obs import obs_blocks
from marl.rl.reward import RewardModule, ball_potential, goal_potential
from marl.rl.rollout import EpisodeRecord, Featurizer
from marl.rl.train import train

SOCCER = {"shared": {"scenario": "soccer"}, "rl": {"reward": {"base": "goal"}, "kick_enabled": True}}


def soccer(shared=None, rl=None, run=None):
    return config_from_dict({"shared": {**SOCCER["shared"], **(shared or {})},
                             "rl": {**SOCCER["rl"], **(rl or {})}, "run": run or {}})


def move(*moves):
    return {"blue": [{"move": list(m), "kick_trigger": 0, "kick_vec": [0.0, 0.0]} for m in moves]}


def play(env, actions, limit=1000):
    for _ in range(limit):
        s, term, trunc, info = env.step(actions)
        if term or trunc:
            return s, term, trunc, info
    raise AssertionError("episode never ended")


@pytest.fixture
def env_1v1():
    env = make_env(soccer())
    env.reset(seed=0)
    yield env
    env.close()


# --- Config and env selection ---------------------------------------------------------------
def test_soccer_needs_the_goal_reward_and_pybullet():
    with pytest.raises(ValueError, match="goal"):
        config_from_dict({"shared": {"scenario": "soccer"}})
    with pytest.raises(NotImplementedError, match="pybullet"):
        soccer(run={"env": "mock"})
    soccer(shared={"n_red": 0})  # an empty net is fine
    soccer(shared={"n_blue": 3, "n_red": 2})


def test_pbrs_ball_to_goal_needs_soccer():
    with pytest.raises(ValueError, match="soccer"):
        config_from_dict({"rl": {"reward": {"shaping": [
            {"name": "pbrs_ball_to_goal", "enabled": True, "coef": 1.0}]}}})


def test_make_env_gives_the_soccer_adapter(env_1v1):
    assert isinstance(env_1v1, PyBulletSoccerEnv)


# --- The adapter: goals and episode ends (ICD §7, §9) ---------------------------------------
def test_blue_goal_ends_the_episode_with_a_goal_event(env_1v1):
    env_1v1.set_state_for_testing(blue_pos=[[6.5, 0.0]], red_pos=[[-6.0, 4.0]], ball_pos=[6.5, 0.0],
                                  holder=("blue", 0))
    s, term, trunc, info = play(env_1v1, move((1.0, 0.0)))
    assert term and not trunc
    assert info["events"][-1] == {"type": "goal", "team": "blue", "scorer": ("blue", 0),
                                  "own_goal": False, "carried": True}
    assert s["ball"]["pos"][0] > 8.0  # over the line, in the pocket


def test_own_goal_is_reds_point(env_1v1):
    env_1v1.set_state_for_testing(blue_pos=[[-6.5, 0.0]], red_pos=[[6.0, 4.0]], ball_pos=[-6.5, 0.0],
                                  holder=("blue", 0))
    _, term, _, info = play(env_1v1, move((-1.0, 0.0)))
    goal = info["events"][-1]
    assert term and goal["team"] == "red" and goal["own_goal"] and goal["scorer"] == ("blue", 0)


def test_red_possession_is_not_terminal(env_1v1):
    env_1v1.set_state_for_testing(blue_pos=[[0.0, 0.0]], red_pos=[[1.5, 0.0]], ball_pos=[0.0, 0.0],
                                  holder=("blue", 0))
    events = []
    for _ in range(20):
        s, term, trunc, info = env_1v1.step(move((0.0, 0.0)))
        events += info["events"]
        assert not term and not trunc
    assert s["red"][0]["has_possession"]
    assert {"type": "possession_change", "from": ("blue", 0), "to": ("red", 0)} in events


def test_time_limit_is_a_draw():
    env = make_env(soccer(shared={"max_episode_steps": 5, "start": {"start_mix": 0.0}}))
    try:
        env.reset(seed=0)
        s, term, trunc, info = play(env, move((0.0, 0.0)))
        assert trunc and not term and s["step"] == 5
        assert not any(e["type"] == "goal" for e in info["events"])
    finally:
        env.close()


# --- Reward ---------------------------------------------------------------------------------
def state(ball, blue=((0.0, 0.0),), red=((3.0, 3.0),), blue_holds=False, red_holds=False):
    return {"step": 0,
            "blue": [{"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": blue_holds and i == 0}
                     for i, p in enumerate(blue)],
            "red": [{"id": j, "pos": list(p), "vel": [0.0, 0.0], "has_possession": red_holds and j == 0}
                    for j, p in enumerate(red)],
            "ball": {"pos": list(ball), "vel": [0.0, 0.0]}}


def goal_info(team):
    return {"events": [{"type": "goal", "team": team, "scorer": None, "own_goal": False,
                        "carried": False}], "kick_applied": [False]}


@pytest.mark.parametrize("team, total", [("blue", 1.0), ("red", -1.0)])
def test_a_goal_reaches_the_learner_as_plus_or_minus_one(team, total):
    cfg = soccer()
    rf = RewardModule(cfg)
    s = state((7.0, 0.0))
    parts = rf.parts(s, {}, s, True, False, goal_info(team), rf.new_episode())
    assert parts.total == pytest.approx(total)
    assert parts.base == pytest.approx(total / (1 - cfg.gamma))
    quiet = rf.parts(s, {}, s, False, False, {"events": [], "kick_applied": [False]}, rf.new_episode())
    assert quiet.total == 0.0


def test_goal_potential_runs_from_about_minus_one_to_zero():
    cfg = soccer()
    assert goal_potential(state((8.0, 0.0)), cfg) == pytest.approx(0.0)
    assert goal_potential(state((-8.0, 0.0)), cfg) == pytest.approx(-1.0)
    assert goal_potential(state((0.0, 0.0)), cfg) > goal_potential(state((-4.0, 0.0)), cfg)


def test_pbrs_ball_to_goal_telescopes():
    cfg = soccer(rl={"reward": {"base": "goal", "shaping": [
        {"name": "pbrs_ball_to_goal", "enabled": True, "coef": 1.0}]}})
    rf, g = RewardModule(cfg), cfg.gamma
    rng = np.random.default_rng(0)
    states = [state(rng.uniform([-8, -5], [8, 5])) for _ in range(30)]
    total = 0.0
    for t, (s, s_next) in enumerate(zip(states, states[1:])):
        last = t == len(states) - 2
        info = goal_info("blue") if last else {"events": [], "kick_applied": [False]}
        total += g**t * rf.parts(s, {}, s_next, last, False, info, rf.new_episode()).shaping
    # terminal: Phi(s_L) = 0, so the shaping sums to -Phi(s_0) for every trajectory
    assert total == pytest.approx(-goal_potential(states[0], cfg), abs=1e-9)


def test_ball_acquisition_counts_a_red_held_ball_only_in_soccer():
    red_holds = state((3.0, 3.0), red_holds=True)
    assert ball_potential(red_holds) == 0.0  # keep-away: anyone holding -> 0
    assert ball_potential(red_holds, red_held_counts=True) == pytest.approx(-math.dist((0, 0), (3, 3)))
    assert ball_potential(state((0, 0), blue_holds=True), red_held_counts=True) == 0.0


def test_memory_keeps_score_only_in_soccer():
    mem = RewardModule(soccer()).new_episode()
    assert mem.outcome == "draw"
    mem.update(goal_info("blue"))
    assert (mem.goals_for, mem.goals_against, mem.outcome) == (1, 0, "win")
    assert RewardModule(config_from_dict({"run": {"env": "mock"}})).new_episode().outcome is None


# --- Observations and warm starts -----------------------------------------------------------
def test_goals_block_only_in_soccer():
    keepaway, socc = Featurizer(config_from_dict({})), Featurizer(soccer())
    assert socc.obs_dim == keepaway.obs_dim + 4
    assert socc.state_dim == keepaway.state_dim
    assert obs_blocks(1, 1)[-1] == ("goals", 4, 0)


def test_goals_block_values():
    f = Featurizer(soccer())
    obs, _ = f(state((0.0, 0.0), blue=((4.0, 2.5),)))
    # attacking goal (8, 0) and defending goal (-8, 0), minus (4, 2.5), over half the field (8, 5)
    np.testing.assert_allclose(obs[0, -4:], [0.5, -0.5, -1.5, -0.5], rtol=1e-6)


@pytest.mark.parametrize("init", ["zero", "random"])
def test_keepaway_actor_warm_starts_into_soccer(init):
    keepaway = config_from_dict({"rl": {"kick_enabled": True}})
    target = soccer(rl={"new_player_init": init})
    torch.manual_seed(0)
    f1 = Featurizer(keepaway)
    actor1, critic1 = build_networks(keepaway, f1.obs_dim, f1.state_dim)
    torch.manual_seed(1)
    f2 = Featurizer(target)
    actor2, critic2 = build_networks(target, f2.obs_dim, f2.state_dim)
    ckpt = {"actor": actor1.state_dict(), "critic": critic1.state_dict(), "config": config_to_dict(keepaway)}
    missing, critic_loaded = warm_start(actor2, critic2, ckpt, target)
    assert missing == [] and not critic_loaded  # a new task: the critic starts fresh
    obs2, _ = f2(state((1.0, -1.0), blue=((2.0, 1.0),), red=((-3.0, 2.0),)))
    w1, w2 = actor1.state_dict()["trunk.0.weight"], actor2.state_dict()["trunk.0.weight"]
    torch.testing.assert_close(w2[:, :-4], w1)
    if init == "zero":
        with torch.no_grad():
            torch.testing.assert_close(actor2.distributions(torch.as_tensor(obs2))[MOVE].mean,
                                       actor1.distributions(torch.as_tensor(obs2[:, :-4]))[MOVE].mean)
    else:
        assert w2[:, -4:].abs().max() > 0


# --- Metrics and baselines ------------------------------------------------------------------
def test_summarize_outcome_rates():
    records = [EpisodeRecord(100, True, False, "free_ball", 1.0, outcome=o)
               for o in ("win", "win", "loss", "draw")]
    out = summarize(records)
    assert (out["win_rate"], out["loss_rate"], out["draw_rate"]) == (0.5, 0.25, 0.25)
    keepaway = summarize([EpisodeRecord(100, True, False, "free_ball", 1.0)])
    assert all(math.isnan(keepaway[k]) for k in ("win_rate", "loss_rate", "draw_rate"))


def test_dribble_baseline_scores_on_an_empty_net():
    cfg = soccer(shared={"n_red": 0, "max_episode_steps": 600})
    assert baselines_for("soccer") == ("random", "still", "dribble")
    env = make_env(cfg)
    try:
        records = run_episodes(env, make_baseline("dribble", cfg.shared), cfg, RewardModule(cfg), range(5))
    finally:
        env.close()
    assert summarize(records)["win_rate"] == 1.0


# --- Training ---------------------------------------------------------------------------------
def test_tiny_soccer_training_run(tmp_path):
    cfg = soccer(shared={"max_episode_steps": 40},
                 rl={"reward": {"base": "goal", "shaping": [
                     {"name": "pbrs_ball_to_goal", "enabled": True, "coef": 1.0},
                     {"name": "pbrs_ball_acquisition", "enabled": True, "coef": 1.0}]},
                     "ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
                 run={"total_steps": 128, "eval_every": 1, "eval_episodes": 2})
    out = train(cfg, tmp_path / "run", log=lambda _msg: None).out_dir
    with open(out / "eval.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and {"win_rate", "loss_rate", "draw_rate"} <= set(rows[0])
    assert (out / "checkpoints" / "final.pt").exists()
### END AI GENERATED
