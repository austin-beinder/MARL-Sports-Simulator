# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-07
# Author:  Austin Beinder
# Usage:
#   - 2026-11-07: Generated this entire file: tests for kicking on the RL side
#     and in the PyBullet adapter (config gating, the kick heads and their
#     masks, the action adapter, a kicking rollout, and training end to end).
# Prompt summary: Wire kicking in: PyBullet game + ICD adapter + config + the
#   actor's kick heads (Bernoulli trigger, Beta kick vector, possession masks).
# Human review:   Pending review
# ==============================================================================
import csv
import math

import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict
from marl.envs.pybullet_env import PyBulletEnv
from marl.rl.actions import actions_from_samples, to_env_actions
from marl.rl.checkpoint import load_actor
from marl.rl.networks import KICK_PRIOR, KICK_TRIGGER, KICK_VEC, MOVE, Actor, build_networks
from marl.rl.obs import OWN_POSSESSION
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer, RolloutCollector, make_buffer
from marl.rl.train import train

OBS_DIM = 17  # 1v1 observation size
SMALL = {
    "rl": {"kick_enabled": True, "ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
    "run": {"env": "pybullet", "total_steps": 128, "eval_every": 2, "eval_episodes": 3},
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def obs_holding(holds: list[bool]) -> torch.Tensor:
    obs = torch.randn(len(holds), OBS_DIM)
    obs[:, OWN_POSSESSION] = torch.tensor([float(h) for h in holds])
    return obs


# --- Config ------------------------------------------------------------------------
def test_kicking_needs_the_pybullet_env():
    config_from_dict({"rl": {"kick_enabled": True}, "run": {"env": "pybullet"}})
    with pytest.raises(NotImplementedError, match="pybullet"):
        config_from_dict({"rl": {"kick_enabled": True}, "run": {"env": "mock"}})


def test_ball_damping_must_not_be_negative():
    with pytest.raises(ValueError, match="ball_damping"):
        config_from_dict({"shared": {"ball_damping": -0.1}})


def test_adapter_passes_kick_speed_and_ball_damping_to_the_game():
    sc = config_from_dict({"shared": {"max_kick_speed": 4.0, "ball_damping": 0.5}}).shared
    env = PyBulletEnv(sc)
    try:
        assert env._game.config.max_kick_speed == 4.0
        assert env._game.config.ball_damping == 0.5
    finally:
        env.close()


# --- PyBullet adapter ------------------------------------------------------------------
def test_adapter_kick_releases_the_ball_and_reports_it():
    sc = config_from_dict({}).shared
    env = PyBulletEnv(sc)
    try:
        env.reset(seed=0)
        env.set_state_for_testing(blue_pos=[[-5, 0]], red_pos=[[6, 4]], ball_pos=[-5, 0], holder=("blue", 0))
        s, term, trunc, info = env.step({"blue": [{"move": [0.0, 0.0], "kick_trigger": 1, "kick_vec": [0.5, 0.0]}]})
        assert info["kick_applied"] == [True]
        assert info["events"][0] == {"type": "kick", "agent": ("blue", 0)}
        assert {"type": "possession_change", "from": ("blue", 0), "to": None} in info["events"]
        assert not s["blue"][0]["has_possession"]
        # kick_vec [0.5, 0] -> 0.5 * max_kick_speed = 3 m/s, minus a little slowdown during the step
        assert 2.5 < s["ball"]["vel"][0] <= 3.0 and abs(s["ball"]["vel"][1]) < 0.05
    finally:
        env.close()


def test_adapter_kick_speed_is_capped_by_the_kick_vec_length():
    sc = config_from_dict({}).shared
    env = PyBulletEnv(sc)
    try:
        env.reset(seed=0)
        env.set_state_for_testing(blue_pos=[[-5, -3]], red_pos=[[6, 4]], ball_pos=[-5, -3], holder=("blue", 0))
        s, *_ = env.step({"blue": [{"move": [0.0, 0.0], "kick_trigger": 1, "kick_vec": [1.0, 1.0]}]})
        speed = math.hypot(*s["ball"]["vel"])
        assert 5.0 < speed <= sc.max_kick_speed  # |kick_vec| = sqrt(2) is scaled back to 1
    finally:
        env.close()


# --- Kick heads ------------------------------------------------------------------------
def test_kick_heads_exist_only_when_enabled():
    assert set(Actor(OBS_DIM).heads) == {MOVE}
    actor = Actor(OBS_DIM, kick_enabled=True)
    assert set(actor.heads) == {MOVE, KICK_TRIGGER, KICK_VEC}
    cfg = config_from_dict(SMALL)
    built, _ = build_networks(cfg, OBS_DIM, 14)
    assert set(built.heads) == {MOVE, KICK_TRIGGER, KICK_VEC}


def test_untrained_trigger_rarely_kicks():
    actor = Actor(OBS_DIM, kick_enabled=True)
    probs = actor.distributions(torch.randn(64, OBS_DIM))[KICK_TRIGGER].probs
    assert torch.allclose(probs, torch.full_like(probs, KICK_PRIOR), atol=0.01)
    samples, _ = actor.act(torch.randn(8, OBS_DIM), deterministic=True)
    assert torch.all(samples[KICK_TRIGGER] == 0.0)


def test_sample_shapes_and_values():
    actor = Actor(OBS_DIM, kick_enabled=True)
    samples, logp = actor.act(obs_holding([True, False, True]))
    assert samples[MOVE].shape == (3, 2) and samples[KICK_VEC].shape == (3, 2)
    assert samples[KICK_TRIGGER].shape == (3, 1) and logp.shape == (3,)
    assert set(samples[KICK_TRIGGER].flatten().tolist()) <= {0.0, 1.0}


def test_masks_count_only_heads_that_affect_the_env():
    actor = Actor(OBS_DIM, kick_enabled=True)
    obs = obs_holding([False, True, True])
    samples = {
        MOVE: torch.full((3, 2), 0.3),
        KICK_TRIGGER: torch.tensor([[1.0], [0.0], [1.0]]),
        KICK_VEC: torch.full((3, 2), 0.8),
    }
    logp, entropy, head_entropy = actor.evaluate(obs, samples)
    d = actor.distributions(obs)
    move = d[MOVE].log_prob(samples[MOVE]).sum(-1)
    trig = d[KICK_TRIGGER].log_prob(samples[KICK_TRIGGER]).sum(-1)
    vec = d[KICK_VEC].log_prob(samples[KICK_VEC]).sum(-1)
    # Not holding: only movement counts, even though the trigger sample is 1
    torch.testing.assert_close(logp[0], move[0])
    # Holding, no kick: movement + trigger
    torch.testing.assert_close(logp[1], move[1] + trig[1])
    # Holding and kicking: all three
    torch.testing.assert_close(logp[2], move[2] + trig[2] + vec[2])
    # Kick-head entropy only counts where the agent holds the ball
    assert head_entropy[KICK_TRIGGER][0] == 0.0 and head_entropy[KICK_VEC][0] == 0.0
    assert head_entropy[KICK_TRIGGER][1] > 0.0
    torch.testing.assert_close(entropy[0], head_entropy[MOVE][0])


def test_rollout_and_update_log_probs_agree():
    # The PPO ratio must start at exactly 1 (the ratio guard checks this every update)
    actor = Actor(OBS_DIM, kick_enabled=True)
    with torch.no_grad():
        actor.heads[KICK_TRIGGER].linear.bias.fill_(0.0)  # kick half the time, so every mask case shows up
    obs = obs_holding([True, False] * 32)
    samples, logp_act = actor.act(obs)
    logp_eval, _, _ = actor.evaluate(obs, samples)
    torch.testing.assert_close(logp_eval, logp_act)


def test_movement_checkpoint_warm_starts_the_kick_actor():
    moving = Actor(OBS_DIM)
    kicking = Actor(OBS_DIM, kick_enabled=True)
    result = kicking.load_state_dict(moving.state_dict(), strict=False)
    assert set(result.missing_keys) == {
        "heads.kick_trigger.linear.weight", "heads.kick_trigger.linear.bias",
        "heads.kick_vec.linear.weight", "heads.kick_vec.linear.bias",
    }
    assert result.unexpected_keys == []
    obs = torch.randn(4, OBS_DIM)
    torch.testing.assert_close(kicking.distributions(obs)[MOVE].mean, moving.distributions(obs)[MOVE].mean)


# --- Action adapter --------------------------------------------------------------------
def test_kick_samples_map_to_icd_actions():
    acts = to_env_actions(np.array([[0.5, 0.5], [0.75, 0.25]]), np.array([[1.0], [0.0]]), np.array([[1.0, 0.5], [0.25, 0.75]]))
    assert acts == {
        "blue": [
            {"move": [0.0, 0.0], "kick_trigger": 1, "kick_vec": [1.0, 0.0]},
            {"move": [0.5, -0.5], "kick_trigger": 0, "kick_vec": [-0.5, 0.5]},
        ]
    }
    assert type(acts["blue"][0]["kick_trigger"]) is int


def test_actions_from_samples_with_and_without_kick_heads():
    for kick_enabled in (False, True):
        actor = Actor(OBS_DIM, kick_enabled=kick_enabled)
        samples, _ = actor.act(obs_holding([True, False]))
        acts = actions_from_samples(samples)
        assert len(acts["blue"]) == 2
        if not kick_enabled:
            assert all(a["kick_trigger"] == 0 and a["kick_vec"] == [0.0, 0.0] for a in acts["blue"])


@pytest.mark.parametrize(
    "trigger, kick",
    [(np.array([1.0]), None), (None, np.array([[0.5, 0.5]])), (np.array([0.5]), np.array([[0.5, 0.5]])),
     (np.array([1.0, 0.0]), np.array([[0.5, 0.5]])), (np.array([1.0]), np.array([[1.5, 0.5]]))],
)
def test_bad_kick_samples_raise(trigger, kick):
    with pytest.raises(ValueError):
        to_env_actions(np.array([[0.5, 0.5]]), trigger, kick)


# --- Rollout and training on PyBullet ----------------------------------------------------
def test_kicking_rollout_on_pybullet():
    cfg = config_from_dict({**SMALL, "shared": {"start": {"start_mix": 1.0}}})  # blue always starts with the ball
    feat = Featurizer(cfg)
    actor, critic = build_networks(cfg, feat.obs_dim, feat.state_dim)
    with torch.no_grad():
        actor.heads[KICK_TRIGGER].linear.bias.fill_(10.0)  # always kick when holding
    env = PyBulletEnv(cfg.shared)
    try:
        col = RolloutCollector(env, cfg, actor, critic, RewardModule(cfg), np.random.default_rng(0))
        buf = make_buffer(cfg, col.featurize, actor)
        stats = col.collect(buf)
    finally:
        env.close()
    assert stats.kicks_per_step > 0
    assert buf.samples[KICK_TRIGGER].shape == (64, 1, 1) and buf.samples[KICK_VEC].shape == (64, 1, 2)


@pytest.mark.parametrize("shared", [{}, {"n_blue": 2, "n_red": 2}])
def test_training_with_kicks_on_pybullet_runs_end_to_end(tmp_path, shared):
    cfg = config_from_dict({**SMALL, "shared": shared})
    res = train(cfg, tmp_path / "kick", log=lambda _m: None)
    assert res.final_eval["episodes"] == 3 and res.final_eval_deterministic["episodes"] == 3
    with open(res.out_dir / "train.csv", newline="") as f:
        assert "kicks_per_step" in next(csv.DictReader(f))
    _, actor, _ = load_actor(res.out_dir / "checkpoints" / "final.pt")  # rebuilds the kick heads
    assert set(actor.heads) == {MOVE, KICK_TRIGGER, KICK_VEC}
