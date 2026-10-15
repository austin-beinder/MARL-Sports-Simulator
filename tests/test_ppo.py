# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: PPO update tests (hand-computed
#     clipped objective and its gradients, advantage normalization, the
#     ratio guard on real and corrupted rollouts, direction tests for the
#     policy loss / entropy bonus / value loss, separate optimizers,
#     determinism, explained variance).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.buffer import RolloutBuffer
from marl.rl.networks import MOVE, Actor, Critic, build_networks
from marl.rl.ppo import PPOUpdater, clipped_surrogate_loss, explained_variance, normalize_advantages
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer, RolloutCollector, make_buffer

OBS = 17


def ppo_cfg(**over):
    ppo = {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 4} | over
    return config_from_dict({"rl": {"ppo": ppo}, "run": {"total_steps": 10_000}}).rl.ppo


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- The clipped objective, by hand ------------------------------------------------------------
def test_clipped_surrogate_by_hand():
    # eps = 0.2. Per sample: min(r*A, clip(r, 0.8, 1.2)*A)
    #   r=1.5, A=+1 -> min(1.5, 1.2)  =  1.2   (clipped: no reward for pushing r further up)
    #   r=0.5, A=+1 -> min(0.5, 0.8)  =  0.5   (unclipped: the pessimistic choice)
    #   r=1.5, A=-1 -> min(-1.5, -1.2) = -1.5  (unclipped: a bad action made MORE likely is fully penalized)
    #   r=0.5, A=-1 -> min(-0.5, -0.8) = -0.8  (clipped: no reward for pushing r further down)
    #   r=1.0, A=+2 -> 2.0
    ratio = torch.tensor([1.5, 0.5, 1.5, 0.5, 1.0], requires_grad=True)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0, 2.0])
    loss = clipped_surrogate_loss(ratio, adv, 0.2)
    assert loss.item() == pytest.approx(-(1.2 + 0.5 - 1.5 - 0.8 + 2.0) / 5)
    loss.backward()
    # Zero gradient exactly where the clip binds: that's PPO's trust region.
    torch.testing.assert_close(ratio.grad, torch.tensor([0.0, -0.2, 0.2, 0.0, -0.4]))


def test_normalize_advantages():
    a = normalize_advantages(torch.tensor([1.0, 2.0, 3.0, 10.0]))
    assert a.mean().item() == pytest.approx(0.0, abs=1e-6) and a.std(correction=0).item() == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(normalize_advantages(torch.full((5,), 3.0)), torch.zeros(5))  # constant -> 0, not NaN
    assert torch.equal(normalize_advantages(torch.tensor([4.0])), torch.zeros(1))  # 1 sample -> 0, not NaN


def test_explained_variance():
    r = np.array([0.0, 1.0, 2.0, 3.0])
    assert explained_variance(r, r) == pytest.approx(1.0)
    assert explained_variance(np.full(4, r.mean()), r) == pytest.approx(0.0)
    assert np.isnan(explained_variance(r, np.ones(4)))


# --- Synthetic buffers: every row its own one-step episode, so A_t = r_t - V_t -----------------------
def synthetic(actor, rewards, xs, T=64):
    buf = RolloutBuffer(T, 1, OBS, OBS, {MOVE: 2})
    obs = torch.randn(T, OBS)
    for t in range(T):
        x = torch.tensor([[xs[t], 0.5]])
        with torch.no_grad():
            lp, _, _ = actor.evaluate(obs[t : t + 1], {MOVE: x})
        buf.add(
            obs=obs[t : t + 1].numpy(),
            critic_in=obs[t : t + 1].numpy(),
            samples={MOVE: x.numpy()},
            log_prob=lp.numpy(),
            value=np.zeros(1),
            next_value=np.zeros(1),
            reward=float(rewards[t]),
            terminated=True,
            truncated=False,
            effective_terminal=True,
        )
    buf.compute_advantages(0.99, 0.95)
    return buf, obs


def mean_x(actor, obs):
    with torch.no_grad():
        return actor.distributions(obs)[MOVE].mean[:, 0].mean().item()


def test_policy_moves_toward_positive_advantage():
    # x = 0.8 was good (+1), x = 0.2 was bad (-1): the update must raise the mean of x.
    actor, critic = Actor(OBS), Critic(OBS)
    good = np.arange(64) % 2 == 0
    buf, obs = synthetic(actor, np.where(good, 1.0, -1.0), np.where(good, 0.8, 0.2))
    before = mean_x(actor, obs)
    PPOUpdater(actor, critic, ppo_cfg(lr=1e-3)).update(buf, np.random.default_rng(0))
    assert mean_x(actor, obs) > before + 0.01


def test_entropy_bonus_raises_entropy_when_advantages_are_zero():
    actor, critic = Actor(OBS), Critic(OBS)
    buf, obs = synthetic(actor, np.zeros(64), np.full(64, 0.5))  # A = 0: only the entropy term acts
    samples = {MOVE: torch.full((64, 2), 0.5)}
    _, h_before, _ = actor.evaluate(obs, samples)
    PPOUpdater(actor, critic, ppo_cfg(ent_coef=0.1, lr=1e-3)).update(buf, np.random.default_rng(0))
    _, h_after, _ = actor.evaluate(obs, samples)
    assert h_after.mean().item() > h_before.mean().item()


def test_critic_regresses_toward_the_returns():
    # Every return is 0.5 while the stored rollout values are 0: the critic must land
    # near 0.5. (A critic fit to the old values would land near 0; measured +0.004.)
    actor, critic = Actor(OBS), Critic(OBS)
    buf, obs = synthetic(actor, np.full(64, 0.5), np.full(64, 0.5))
    PPOUpdater(actor, critic, ppo_cfg(lr=1e-3, update_epochs=10)).update(buf, np.random.default_rng(0))
    with torch.no_grad():
        v = critic(obs)
    assert abs(v.mean().item() - 0.5) < 0.1 and ((v - 0.5) ** 2).mean().item() < 0.05  # measured 0.490, 0.007


def test_actor_update_ignores_the_advantage_scale():
    # Per-minibatch normalization: rewards x100 must give the SAME policy update (the
    # critic still sees the scaled returns). Without normalization the steps would differ.
    good = np.arange(64) % 2 == 0
    xs, r = np.where(good, 0.8, 0.2), np.where(good, 1.0, -1.0)
    params = []
    for scale in (1.0, 100.0):
        torch.manual_seed(0)
        actor, critic = Actor(OBS), Critic(OBS)
        buf, _ = synthetic(actor, scale * r, xs)
        PPOUpdater(actor, critic, ppo_cfg()).update(buf, np.random.default_rng(0))
        params.append([p.detach().clone() for p in actor.parameters()])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(*params))


def test_gradient_clipping_is_applied():
    # Adam undoes gradient SCALE, but gradients clipped far below its eps (1e-5)
    # barely move the weights, so a tiny max_grad_norm must freeze the actor.
    torch.manual_seed(0)
    actor, critic = Actor(OBS), Critic(OBS)
    good = np.arange(64) % 2 == 0
    buf, _ = synthetic(actor, np.where(good, 1.0, -1.0), np.where(good, 0.8, 0.2))
    before = [p.detach().clone() for p in actor.parameters()]
    stats = PPOUpdater(actor, critic, ppo_cfg(max_grad_norm=1e-9)).update(buf, np.random.default_rng(0))
    assert max((a - b).abs().max().item() for a, b in zip(before, actor.parameters())) < 1e-6
    assert stats["grad_norm_actor"] > 1e-3  # the logged norm is the pre-clip norm


# --- The ratio guard (walk-through step 4) -------------------------------------------------------------
def real_rollout(T=64, seed=0):
    cfg = config_from_dict({"rl": {"ppo": {"rollout_steps": T, "num_minibatches": 4, "update_epochs": 2}}, "run": {"total_steps": 10_000}})
    torch.manual_seed(seed)
    f = Featurizer(cfg)
    actor, critic = build_networks(cfg, f.obs_dim, f.state_dim)
    col = RolloutCollector(MockEnv(cfg.shared), cfg, actor, critic, RewardModule(cfg), np.random.default_rng(seed))
    buf = make_buffer(cfg, f, actor)
    col.collect(buf)
    buf.compute_advantages(cfg.gamma, cfg.rl.ppo.gae_lambda)
    return cfg, actor, critic, buf


def test_ratio_is_one_on_the_first_minibatch_of_a_real_rollout():
    cfg, actor, critic, buf = real_rollout()
    stats = PPOUpdater(actor, critic, cfg.rl.ppo).update(buf, np.random.default_rng(0))
    assert stats["ratio_check"] < 1e-6
    assert stats["updates"] == cfg.rl.ppo.update_epochs * cfg.rl.ppo.num_minibatches
    expected = {"policy_loss", "value_loss", "entropy", "entropy_move", "approx_kl", "clip_frac",
                "grad_norm_actor", "grad_norm_critic", "ratio_check", "updates"}
    assert set(stats) == expected and all(np.isfinite(v) for v in stats.values())


def test_ratio_guard_stops_training_on_inconsistent_log_probs():
    cfg, actor, critic, buf = real_rollout()
    buf.log_probs += 0.01  # e.g. log-probs stored from a different sample than the one stored
    before = [p.detach().clone() for p in actor.parameters()]
    with pytest.raises(RuntimeError, match="ratio guard"):
        PPOUpdater(actor, critic, cfg.rl.ppo).update(buf, np.random.default_rng(0))
    assert all(torch.equal(a, b) for a, b in zip(before, actor.parameters()))  # nothing was updated


# --- Structure and reproducibility ------------------------------------------------------------------
def test_actor_and_critic_have_separate_optimizers():
    actor, critic = Actor(OBS), Critic(OBS)
    up = PPOUpdater(actor, critic, ppo_cfg())
    a_ids = {id(p) for g in up.actor_opt.param_groups for p in g["params"]}
    c_ids = {id(p) for g in up.critic_opt.param_groups for p in g["params"]}
    assert a_ids == {id(p) for p in actor.parameters()} and c_ids == {id(p) for p in critic.parameters()}
    assert not a_ids & c_ids


def test_update_is_deterministic():
    results = []
    for _ in range(2):
        cfg, actor, critic, buf = real_rollout(seed=1)
        PPOUpdater(actor, critic, cfg.rl.ppo).update(buf, np.random.default_rng(5))
        results.append([p.detach().clone() for p in [*actor.parameters(), *critic.parameters()]])
    assert all(torch.equal(a, b) for a, b in zip(*results))
