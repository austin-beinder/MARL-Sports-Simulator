# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-14
# Author:  Austin Beinder
# Usage:
#   - 2026-11-14: Generated this entire file: tests for the call head's own
#     per-blue reward (call_values, the buffer's call advantages, the PPO
#     update's separate call surrogate), the claim_ball scenario (spawn, both
#     envs, config, the claim reward) and the call_yield shaping term.
#   - 2026-11-14: The dead_ball shaping term, receiver_spacing leaving out the
#     blue nearest a dead ball, and the catch stages' new terms.
# Prompt summary: Give the "got it" call its own per-blue reward instead of the
#   shared team reward, and add a claim_ball stage where three blues claim a
#   loose ball: pay a blue for calling a ball it then gets, and charge blues
#   who crowd a ball a teammate called.
# Human review:   Pending review
# ==============================================================================
"""The "got it" call's own reward, and the claim_ball scenario that practises it (2026-11-14)."""

import csv
import math

import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict, with_overrides
from marl.common.spawn import START_FREE_BALL, sample_spawn
from marl.envs.mock_env import MockEnv
from marl.envs.pybullet_env import PyBulletEnv
from marl.rl.buffer import RolloutBuffer
from marl.rl.networks import CALL, MOVE, Actor, Critic
from marl.rl.ppo import PPOUpdater
from marl.rl.reward import (CALL_STEP_COST, CALL_WRONG, DEAD_BALL_SPEED, YIELD_DISTANCE, YIELD_STEAL, RewardModule,
                            blue_pickup, call_values, call_yield, claim, dead_ball, receiver_spacing)
from marl.rl.train import train

CLAIM = {
    "shared": {"scenario": "claim_ball", "n_blue": 3, "n_red": 0, "max_episode_steps": 100},
    "rl": {"call_enabled": True, "reward": {"base": "claim", "shaping": [
        {"name": "call_accuracy", "enabled": True, "coef": 1.0},
        {"name": "call_yield", "enabled": True, "coef": 0.2}]}},
}


def caught_by(i):
    return {"type": "possession_change", "from": None, "to": ("blue", i)}


# --- Scoring calls in hindsight -----------------------------------------------------------
def test_call_values_score_the_next_pickup():
    q, valid = call_values([-1, -1, -1, 1, -1], [False] * 5, n_agents=3, lead=2, cost=0.0)
    assert q[0].tolist() == [0.0, 0.0, 0.0]  # nobody got the ball within 2 steps
    assert q[1].tolist() == q[2].tolist() == [CALL_WRONG, 1.0, CALL_WRONG]  # blue 1 got it: its call was right
    assert valid[:3].all() and not valid[3:].any()  # the last two windows run past the rollout
    q_cost, _ = call_values([-1, -1, -1, 1, -1], [False] * 5, n_agents=3, lead=2)
    assert np.allclose(q_cost, q - CALL_STEP_COST)  # calling at all costs a little


def test_call_values_stop_at_the_episode_end():
    q, valid = call_values([-1, -1, 2], [False, True, False], n_agents=3, lead=3, cost=0.0)
    assert not q[:2].any() and valid[:2].all()  # the pickup in step 2 belongs to the next episode
    assert not valid[2].any()


def test_a_call_counts_only_from_the_next_step():
    q, _ = call_values([1, -1, -1], [False] * 3, n_agents=2, lead=1, cost=0.0)
    assert q[0].tolist() == [0.0, 0.0]  # a pickup in the same step as the call: teammates couldn't see it


def test_blue_pickup_reads_the_events():
    assert blue_pickup({"events": []}) is None
    assert blue_pickup({"events": [{"type": "kick", "agent": ("blue", 0)}, caught_by(2)]}) == 2
    red = {"type": "possession_change", "from": None, "to": ("red", 0)}
    assert blue_pickup({"events": [red]}) is None


def test_the_buffer_scores_each_blues_call():
    buf = RolloutBuffer(4, 2, 3, 3, {MOVE: 2, CALL: 1})
    p0, p1 = 0.25, 0.25  # both blues call with probability 0.25; blue 0 calls, blue 1 doesn't
    lp = np.log([p0, 1 - p1])
    for t in range(4):
        buf.add(obs=np.zeros((2, 3)), critic_in=np.zeros((2, 3)), samples={MOVE: np.full((2, 2), 0.5),
                CALL: np.array([[1.0], [0.0]])}, log_prob=lp, value=np.zeros(2), next_value=np.zeros(2),
                reward=0.0, terminated=False, truncated=False, effective_terminal=False,
                pickup=0 if t == 2 else None, call_log_prob=lp)
    buf.compute_advantages(0.99, 0.95)
    buf.compute_call_advantages()
    right, wrong = 1.0 - CALL_STEP_COST, CALL_WRONG - CALL_STEP_COST
    # A = q * (a - p): blue 0 called and was right; blue 1 rightly stayed quiet (calling was worth less)
    assert buf.call_advantages[0] == pytest.approx([right * (1 - p0), wrong * (0 - p1)])
    assert buf.call_advantages[0, 1] > 0
    assert buf.call_valid[:2].all() and not buf.call_valid[2:].any()


# --- The PPO update ----------------------------------------------------------------------
def call_update(pickup):
    """One PPO update on a 64-step rollout in which ``pickup`` gets the ball every step (None: nobody
    does). Returns the mean chance of calling before and after, and the update's stats."""
    torch.manual_seed(0)
    ppo = config_from_dict({"rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 4,
                                           "lr": 1e-2}},  # big steps, so the change shows
                            "run": {"total_steps": 10_000}}).rl.ppo
    actor, critic = Actor(5, call_enabled=True), Critic(5)
    obs = torch.randn(64, 5)
    buf = RolloutBuffer(64, 1, 5, 5, {name: head.action_dim for name, head in actor.heads.items()})
    for t in range(64):
        samples, logp = actor.act(obs[t : t + 1])
        buf.add(obs=obs[t : t + 1], critic_in=obs[t : t + 1], samples={h: x.numpy() for h, x in samples.items()},
                log_prob=logp, value=critic(obs[t : t + 1]).detach(), next_value=torch.zeros(1), reward=0.0,
                terminated=False, truncated=False, effective_terminal=False, pickup=pickup,
                call_log_prob=actor.head_log_prob(obs[t : t + 1], samples, CALL))
    buf.compute_advantages(0.99, 0.95)
    buf.compute_call_advantages()
    before = actor.distributions(obs)[CALL].probs.mean().item()
    stats = PPOUpdater(actor, critic, ppo, call_coef=1.0).update(buf, np.random.default_rng(0))
    return before, actor.distributions(obs)[CALL].probs.mean().item(), stats


def test_the_call_head_learns_from_its_own_advantage():
    before, after, stats = call_update(pickup=0)  # blue 0 gets the ball every step: calling is right
    assert stats["ratio_check"] < 1e-6 and math.isfinite(stats["call_loss"])
    assert after > before + 0.01
    before, after, _ = call_update(pickup=None)  # nobody ever gets it: calling only costs
    assert after < before - 0.01


def test_evaluate_split_adds_up_to_evaluate():
    actor = Actor(5, call_enabled=True, kick_enabled=True)
    obs = torch.randn(8, 5)
    samples, _ = actor.act(obs)
    logp, _, _ = actor.evaluate(obs, samples)
    rest, call, _, _ = actor.evaluate_split(obs, samples)
    assert torch.allclose(rest + call, logp, atol=1e-6)
    assert torch.allclose(call, actor.head_log_prob(obs, samples, CALL))


def test_call_accuracy_sets_the_call_coef_not_the_team_reward():
    rm = RewardModule(config_from_dict(CLAIM))
    assert rm.call_coef == 1.0
    without = RewardModule(config_from_dict({**CLAIM, "rl": {"call_enabled": True, "reward": {"base": "claim"}}}))
    assert without.call_coef == 0.0


# --- claim_ball ---------------------------------------------------------------------------
def state(holder=None, ball=(0.5, 0.0), calling=(), blues=((0.0, 0.0), (3.0, 0.0), (0.0, 3.0))):
    return {"step": 0, "red": [],
            "blue": [{"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": i == holder,
                      "calling": i in calling} for i, p in enumerate(blues)],
            "ball": {"pos": list(blues[holder] if holder is not None else ball), "vel": [0.0, 0.0]}}


def yield_term(s, s_next, events=()):
    return call_yield(s, {}, s_next, False, False, {"events": list(events)}, None)


def test_call_yield_charges_blues_crowding_a_called_ball():
    s = state(calling={1})  # blue 1 called; blue 0 stands 0.5 m from the ball
    assert yield_term(s, s) == pytest.approx(-(1 - 0.5 / YIELD_DISTANCE))
    assert yield_term(state(calling={0}), state(calling={0})) == 0.0  # the caller may stand on it
    assert yield_term(state(), state()) == 0.0  # nobody called
    assert yield_term(s, state(0, calling={1}), [caught_by(0)]) == -YIELD_STEAL  # took a called ball
    assert yield_term(s, state(1, calling={1}), [caught_by(1)]) == 0.0  # the caller got it
    assert yield_term(state(2, calling={1}), state(2, calling={1})) == 0.0  # not loose: nothing to leave


def test_claim_costs_every_step_until_a_blue_has_the_ball():
    assert claim(state(), {}, state(), False, False, {}, None) == -1.0
    assert claim(state(), {}, state(1), True, False, {}, None) == 0.0


# --- claim's urgency in the catch stages (2026-11-14) ----------------------------------------
def test_dead_ball_charges_a_stopped_loose_ball_only():
    def term(s_next):
        return dead_ball(None, {}, s_next, False, False, {}, None)

    assert term(state()) == -1.0  # loose and still
    assert term(state(1)) == 0.0  # held
    rolling = state()
    rolling["ball"]["vel"] = [DEAD_BALL_SPEED + 0.5, 0.0]  # a pass in flight is free
    assert term(rolling) == 0.0


def test_receiver_spacing_leaves_out_the_blue_fetching_a_dead_ball():
    cfg = config_from_dict({**CLAIM, "rl": {**CLAIM["rl"], "reward": {"base": "claim", "spacing_distance": 3.0}}})

    def spacing(s_next):
        return receiver_spacing(None, {}, s_next, False, False, {}, cfg)

    blues = ((0.0, 0.0), (3.0, 0.0), (9.0, 0.0))  # pairs 3, 9 and 6 m apart: only 0-1 pays
    assert spacing(state(1, blues=blues)) == pytest.approx(1 / 3)  # held: every pair counts
    assert spacing(state(ball=(9.5, 0.0), blues=blues)) == pytest.approx(1.0)  # 2 fetches: 0-1 left
    assert spacing(state(ball=(0.5, 0.0), blues=blues)) == pytest.approx(0.0)  # 0 fetches: 1-2, 6 m
    rolling = state(ball=(9.5, 0.0), blues=blues)
    rolling["ball"]["vel"] = [2.0, 0.0]  # a pass in flight: every pair counts
    assert spacing(rolling) == pytest.approx(1 / 3)


def test_claim_ball_always_starts_loose():
    cfg = config_from_dict(CLAIM)
    opts = {**cfg.shared.start.to_reset_options(), "start_mix": 1.0}  # would be all possession starts in keepaway
    for seed in range(20):
        sp = sample_spawn(np.random.default_rng(seed), scenario="claim_ball", n_blue=3, n_red=0,
                          field_size=cfg.shared.field_size, player_radius=cfg.shared.player_radius,
                          possession_radius=cfg.shared.possession_radius, options=opts)
        assert sp.start_type == START_FREE_BALL and sp.holder is None


def test_claim_ball_config_checks():
    with pytest.raises(ValueError, match="n_red = 0"):
        config_from_dict({**CLAIM, "shared": {**CLAIM["shared"], "n_red": 1}})
    with pytest.raises(ValueError, match="needs rl.reward.base = 'claim'"):
        config_from_dict({**CLAIM, "rl": {"reward": {"base": "survival"}}})
    with pytest.raises(ValueError, match="call_yield shaping term needs rl.call_enabled"):
        config_from_dict({**CLAIM, "rl": {"reward": {"base": "claim", "shaping": [
            {"name": "call_yield", "enabled": True, "coef": 0.2}]}}})


@pytest.mark.parametrize("make", [MockEnv, PyBulletEnv], ids=["mock", "pybullet"])
def test_claim_ball_ends_when_a_blue_gets_the_ball(make):
    cfg = config_from_dict({**CLAIM, "run": {"env": "pybullet" if make is PyBulletEnv else "mock"}})
    env = make(cfg.shared)
    try:
        s, _ = env.reset(seed=0)
        assert not any(b["has_possession"] for b in s["blue"])
        env.set_state_for_testing(blue_pos=[[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]], ball_pos=[1.0, 0.0], holder=None)
        go = {"blue": [{"move": [1.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}]
              + [{"move": [0.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}] * 2}
        for _ in range(20):
            s, terminated, truncated, info = env.step(go)
            if terminated:
                break
        assert terminated and not truncated and s["blue"][0]["has_possession"]
        assert blue_pickup(info) == 0
    finally:
        if hasattr(env, "close"):
            env.close()


def test_claim_ball_trains_end_to_end(tmp_path):
    cfg = with_overrides(config_from_dict(CLAIM), {
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
    })
    result = train(cfg, tmp_path / "claim", log=lambda *_: None)
    assert result.iterations == 2
    with open(tmp_path / "claim" / "train.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(math.isfinite(float(r["call_loss"])) for r in rows)
