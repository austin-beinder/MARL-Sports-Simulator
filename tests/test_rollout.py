# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: rollout collector tests that
#     replay a recorded env log and check every stored row (values,
#     next_value = V(true s_{t+1}) at episode ends, rewards, flags, episode
#     records), episodes spanning rollouts, value-cache refresh after an
#     update, determinism, goto_target and N=2/M=2.
#   - 2026-10-11: Added a check of the episode mean_reward against the replay.
#   - 2026-10-14: Added a test that the shaping reward is reported separately
#     and never enters an episode's mean_reward (PBRS ablation).
# Prompt summary: Implement Phase 3 step 7 of the issue #7 plan (buffer +
#   GAE, plus the rollout collector that owns the next_value contract).
# Human review:   Pending review
# ==============================================================================
import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict
from marl.common.spawn import START_FREE_BALL, START_POSSESSION, START_TARGET
from marl.envs.mock_env import MockEnv
from marl.rl.networks import build_networks
from marl.rl.reward import RewardModule, effective_terminal
from marl.rl.rollout import Featurizer, RolloutCollector, make_buffer, start_type_of

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


class RecordingEnv:
    """Wraps an env and logs every reset and step, to replay what the collector saw."""

    def __init__(self, env):
        self.env, self.log = env, []

    def reset(self, seed, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        self.log.append(("reset", s))
        return s, info

    def step(self, actions):
        out = self.env.step(actions)
        self.log.append(("step", actions, out))
        return out


def transitions(log):
    """(s, actions, s_next, terminated, truncated, info) for every step, in order."""
    out, s = [], None
    for entry in log:
        if entry[0] == "reset":
            s = entry[1]
        else:
            _, actions, (s_next, term, trunc, info) = entry
            out.append((s, actions, s_next, term, trunc, info))
            s = s_next
    return out


def setup(data=None, T=300, seed=0):
    data = dict(data or {})
    rl = dict(data.get("rl", {}))
    rl["ppo"] = {"rollout_steps": T, "num_minibatches": 1}
    data["rl"] = rl
    data["run"] = {"total_steps": 10_000}
    cfg = config_from_dict(data)
    torch.manual_seed(seed)
    feat = Featurizer(cfg)
    actor, critic = build_networks(cfg, feat.obs_dim, feat.state_dim)
    env = RecordingEnv(MockEnv(cfg.shared))
    col = RolloutCollector(env, cfg, actor, critic, RewardModule(cfg), np.random.default_rng(seed))
    return cfg, env, col, make_buffer(cfg, col.featurize, actor), critic


def V(critic, featurize, s):
    with torch.no_grad():
        return critic(torch.from_numpy(featurize(s)[1])).numpy()


def test_every_stored_row_matches_a_replay_of_the_env_log():
    cfg, env, col, buf, critic = setup({"shared": {"max_episode_steps": 60}}, T=300)
    stats = col.collect(buf)
    steps = transitions(env.log)
    assert len(steps) == 300 and buf.full
    ends = 0
    for t, (s, actions, s_next, term, trunc, info) in enumerate(steps):
        obs, cin = col.featurize(s)
        np.testing.assert_array_equal(buf.obs[t], obs)
        np.testing.assert_array_equal(buf.critic_in[t], cin)
        np.testing.assert_allclose(buf.values[t], V(critic, col.featurize, s), rtol=1e-6, atol=1e-7)
        # next_value = V(TRUE s_{t+1}): the final state at an episode end, never the reset state.
        np.testing.assert_allclose(buf.next_values[t], V(critic, col.featurize, s_next), rtol=1e-6, atol=1e-7)
        eff = effective_terminal(term, trunc, cfg.rl.horizon_mode)
        assert buf.rewards[t] == pytest.approx(col.reward_fn(s, actions, s_next, eff, trunc, info))
        assert (buf.terminated[t], buf.truncated[t], buf.effective_terminal[t]) == (term, trunc, eff)
        np.testing.assert_allclose(2 * buf.samples["move"][t] - 1, [a["move"] for a in actions["blue"]], atol=1e-6)
        if term or trunc:
            ends += 1
            if t + 1 < 300:
                assert steps[t + 1][0]["step"] == 0  # the next row starts a fresh episode
    assert ends == len(stats.episodes)
    assert any(e.terminated for e in stats.episodes) and any(e.truncated for e in stats.episodes)
    end_states = [s_next for (_, _, s_next, term, trunc, _) in steps if term or trunc]
    assert [e.length for e in stats.episodes] == [s["step"] for s in end_states]
    assert all(e.length <= 60 for e in stats.episodes)
    assert 0.0 <= stats.mean_commanded_speed <= 1.0


def test_episode_spans_rollouts_and_a_rollout_cut_is_not_a_truncation():
    cfg, env, col, buf, critic = setup(T=10)
    stats = col.collect(buf)
    # Precondition: the first rollout ended mid-episode (red needs > 10 steps here).
    assert not (buf.terminated[-1] or buf.truncated[-1])
    assert stats.episodes == []  # the rollout cut is NOT logged as an episode end
    last_next = transitions(env.log)[-1][2]
    col.collect(buf)
    steps = transitions(env.log)
    assert steps[10][0] == last_next  # rollout 2 continues the same episode, no reset
    assert sum(1 for e in env.log if e[0] == "reset") == 1


def test_value_cache_is_refreshed_after_the_critic_changes():
    cfg, env, col, buf, critic = setup(T=10)
    col.collect(buf)
    assert not (buf.terminated[-1] or buf.truncated[-1])  # the next rollout reuses this state
    with torch.no_grad():
        critic.v.bias += 1.0  # stands in for a PPO update
    col.collect(buf)
    with torch.no_grad():
        expected = critic(torch.from_numpy(buf.critic_in[0])).numpy()
    np.testing.assert_allclose(buf.values[0], expected, rtol=1e-6)  # not the stale pre-update value


def test_same_seeds_same_rollout():
    # Sampling uses torch's GLOBAL RNG, so each run must seed and finish before the next
    # starts (interleaving two collectors would share one random stream).
    def run(seed):
        cfg, env, col, buf, critic = setup(T=120, seed=seed)
        col.collect(buf)
        return buf

    a, b = run(3), run(3)
    for field in ("obs", "log_probs", "values", "next_values", "rewards", "terminated", "truncated"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.samples["move"], b.samples["move"])
    assert not np.array_equal(run(4).obs, a.obs)


def test_goto_target_rollout():
    cfg, env, col, buf, critic = setup({**GOTO, "shared": {**GOTO["shared"], "max_episode_steps": 20}}, T=64)
    stats = col.collect(buf)
    assert not buf.terminated.any() and not buf.effective_terminal.any()
    assert [e.truncated for e in stats.episodes] == [True, True, True]
    assert all(e.length == 20 and e.start_type == START_TARGET for e in stats.episodes)
    assert np.all(buf.rewards < 0)  # -(1 - gamma) * distance
    # mean_reward = minus the mean blue-to-target distance over the episode's steps (meters)
    steps = transitions(env.log)
    for k, ep in enumerate(stats.episodes):
        dists = [np.hypot(*np.subtract(s1["blue"][0]["pos"], s1["ball"]["pos"])) for (_, _, s1, *_) in steps[20 * k : 20 * (k + 1)]]
        assert ep.mean_reward == pytest.approx(-np.mean(dists))


def test_n2_m2_rollout_shapes_and_shared_critic_input():
    cfg, env, col, buf, critic = setup({"shared": {"n_blue": 2, "n_red": 2}}, T=40)
    col.collect(buf)
    assert buf.obs.shape == (40, 2, 29) and buf.critic_in.shape == (40, 2, 24)
    assert np.array_equal(buf.critic_in[:, 0], buf.critic_in[:, 1])  # global: same s for both agents
    assert np.array_equal(buf.values[:, 0], buf.values[:, 1])  # -> same V (#5)
    assert not np.array_equal(buf.obs[:, 0], buf.obs[:, 1])  # but different self-first o_i
    buf.compute_advantages(cfg.gamma, cfg.rl.ppo.gae_lambda)
    assert np.array_equal(buf.advantages[:, 0], buf.advantages[:, 1])


def test_shaping_is_reported_separately_and_kept_out_of_mean_reward():
    free_ball = {"start": {"start_mix": 0.0}, "max_episode_steps": 60}
    shaping = [{"name": "pbrs_ball_acquisition", "enabled": True, "coef": 2.0}]
    cfg, env, col, buf, critic = setup({"shared": free_ball, "rl": {"reward": {"shaping": shaping}}}, T=300)
    stats = col.collect(buf)
    # The survival base is 1 per step, so everything above 1 in the stored rewards is shaping.
    assert stats.mean_shaping == pytest.approx(float(buf.rewards.mean()) / col.reward_fn.scale - 1.0, abs=1e-6)
    assert stats.mean_shaping != 0.0
    assert stats.episodes and all(e.mean_reward == 1.0 for e in stats.episodes)  # the task reward only

    cfg, env, col, buf, critic = setup({"shared": free_ball}, T=60)
    assert col.collect(buf).mean_shaping == 0.0  # exactly 0 without shaping terms


def test_start_types_are_inferred_from_the_reset_state():
    cfg, env, col, buf, critic = setup({"shared": {"max_episode_steps": 15}}, T=300)
    stats = col.collect(buf)
    assert {e.start_type for e in stats.episodes} == {START_POSSESSION, START_FREE_BALL}
    s = {"blue": [{"has_possession": True}]}
    assert start_type_of(s, "keepaway") == START_POSSESSION
    assert start_type_of({"blue": [{"has_possession": False}]}, "keepaway") == START_FREE_BALL
    assert start_type_of(s, "goto_target") == START_TARGET
