# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: tests for the actor/critic and the
#     Beta movement head: init, clamping, rollout-vs-update log-prob
#     consistency, analytic entropy (incl. a demonstration of why the
#     sample-based estimate fails), finite gradients at extremes, detached
#     rollout outputs, critic inputs at N=2/M=2, and the warm-start layout.
# Prompt summary: Implement Phase 3 step 6 of the issue #7 plan (actor/critic
#   + Beta movement head + action adapter).
# Human review:   Pending review
# ==============================================================================
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.networks import MOVE, Actor, Critic, build_networks, critic_input, critic_input_dim
from marl.rl.obs import ObsBuilder
from marl.rl.state import StateFlattener, parse_state

OBS_DIM = 17  # 1v1 observation size
EPS = 1e-6


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def force_head(actor, alpha_raw, beta_raw):
    """Pin the movement head's raw outputs (zero weights, chosen biases)."""
    lin = actor.heads[MOVE].linear
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.tensor([*alpha_raw, *beta_raw], dtype=torch.float32))


# --- Initialization ---------------------------------------------------------------
def test_initial_policy_is_wide_and_centered():
    actor = Actor(OBS_DIM)
    obs = torch.randn(512, OBS_DIM)
    d = actor.distributions(obs)[MOVE]
    start = 1.0 + math.log(2.0)  # softplus(0) + 1
    for conc in (d.concentration1, d.concentration0):  # alpha, beta
        torch.testing.assert_close(conc, torch.full_like(conc, start), atol=0.03, rtol=0)
    samples, _ = actor.act(obs, deterministic=True)
    assert (2 * samples[MOVE] - 1).abs().max() < 0.02  # mean action ~ stand still


def test_deterministic_action_is_the_beta_mean():
    # alpha = 3, beta = 2: mean 3/5 = 0.6 (the mode would be 2/3).
    actor = Actor(OBS_DIM)
    r3, r2 = math.log(math.e**2 - 1), math.log(math.e - 1)  # softplus^-1(2), softplus^-1(1)
    force_head(actor, alpha_raw=(r3, r3), beta_raw=(r2, r2))
    samples, _ = actor.act(torch.randn(4, OBS_DIM), deterministic=True)
    torch.testing.assert_close(samples[MOVE], torch.full((4, 2), 0.6))


def test_initial_entropy_is_near_the_uniform_maximum():
    # Beta entropy on [0, 1] is <= 0, with its max (0) at the uniform distribution.
    actor = Actor(OBS_DIM)
    obs = torch.randn(64, OBS_DIM)
    samples, _ = actor.act(obs)
    _, entropy, heads = actor.evaluate(obs, samples)
    assert set(heads) == {MOVE}
    assert -0.3 < entropy.mean().item() < 0.0


def test_identical_obs_give_identical_distributions():
    # Parameter sharing with no agent ID: the policy is a function of o_i only (#4).
    actor = Actor(OBS_DIM)
    obs = torch.randn(1, OBS_DIM).repeat(3, 1)
    d = actor.distributions(obs)[MOVE]
    torch.testing.assert_close(d.concentration1[0], d.concentration1[2])
    torch.testing.assert_close(d.concentration0[0], d.concentration0[2])


# --- Clamping ------------------------------------------------------------------------
def test_why_clamp_a_boundary_sample_gives_infinite_log_prob():
    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    lp = Beta(a, b).log_prob(torch.tensor(1.0))  # (b - 1) * log(1 - 1) = -inf
    lp.backward()
    assert torch.isinf(lp) and torch.isinf(b.grad)
    lp_clamped = Beta(a, b).log_prob(torch.tensor(1.0 - EPS))
    assert torch.isfinite(lp_clamped)


def test_samples_are_clamped_at_extreme_parameters():
    actor = Actor(OBS_DIM, beta_eps=EPS)
    # x: Beta(~1e6, 1) piles up near 1; y: Beta(1, ~1e6) piles up near 0.
    force_head(actor, alpha_raw=(1e6, -30.0), beta_raw=(-30.0, 1e6))
    obs = torch.randn(5000, OBS_DIM)
    samples, logp = actor.act(obs)
    x = samples[MOVE]
    lo, hi = torch.tensor(EPS), torch.tensor(1.0 - EPS)
    assert (x >= lo).all() and (x <= hi).all()
    assert (x[:, 0] == hi).any() and (x[:, 1] == lo).any()  # the clamp really was exercised
    assert torch.isfinite(logp).all()
    # The rollout log-prob must come from the CLAMPED (stored) sample: here the clamp
    # changes values, so a log-prob of the raw sample would disagree with the update.
    logp_update, _, _ = actor.evaluate(obs, samples)
    torch.testing.assert_close(logp, logp_update, atol=1e-6, rtol=1e-6)


def test_losses_and_gradients_finite_at_extremes():
    actor = Actor(OBS_DIM)
    force_head(actor, alpha_raw=(1e6, -30.0), beta_raw=(-30.0, 1e6))
    obs = torch.randn(256, OBS_DIM)
    samples, logp_old = actor.act(obs)
    logp, entropy, _ = actor.evaluate(obs, samples)
    adv = torch.randn(256)
    ratio = torch.exp(logp - logp_old)
    loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean() - 0.001 * entropy.mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in actor.named_parameters():
        assert torch.isfinite(p.grad).all(), name


# --- Log-prob consistency (the PPO ratio starts at exactly 1) ------------------------------
def test_log_prob_identical_at_rollout_and_update():
    actor = Actor(OBS_DIM)
    obs = torch.randn(64, OBS_DIM)
    # The rollout acts one control step at a time...
    rows = [actor.act(obs[t : t + 1]) for t in range(64)]
    stored = torch.cat([r[0][MOVE] for r in rows])
    logp_rollout = torch.cat([r[1] for r in rows])
    # ...and the update re-scores the stored samples in a big batch.
    logp_update, _, _ = actor.evaluate(obs, {MOVE: stored})
    torch.testing.assert_close(logp_update, logp_rollout, atol=1e-6, rtol=1e-6)
    # Minibatches shuffle and subsample rows; still the same numbers.
    idx = torch.randperm(64)[:16]
    logp_mb, _, _ = actor.evaluate(obs[idx], {MOVE: stored[idx]})
    torch.testing.assert_close(logp_mb, logp_rollout[idx], atol=1e-6, rtol=1e-6)


def test_rollout_outputs_carry_no_gradient():
    # logp_old must be a constant in the ratio; a graph here would corrupt the PPO gradient.
    actor = Actor(OBS_DIM)
    samples, logp = actor.act(torch.randn(4, OBS_DIM))
    assert not samples[MOVE].requires_grad and not logp.requires_grad


def test_evaluate_rejects_mismatched_heads():
    actor = Actor(OBS_DIM)
    with pytest.raises(ValueError):
        actor.evaluate(torch.randn(2, OBS_DIM), {"kick_vec": torch.full((2, 2), 0.5)})


# --- Analytic entropy (#10, #18) ----------------------------------------------------------
def test_entropy_is_the_closed_form_and_equals_expected_neg_log_prob():
    a = torch.tensor([3.0, 1.69, 10.0], dtype=torch.float64)
    b = torch.tensor([2.0, 1.69, 1.5], dtype=torch.float64)
    closed = (
        torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        - (a - 1) * torch.digamma(a) - (b - 1) * torch.digamma(b) + (a + b - 2) * torch.digamma(a + b)
    )
    torch.testing.assert_close(Beta(a, b).entropy(), closed)
    x = Beta(a, b).sample((200_000,))
    monte_carlo = -Beta(a, b).log_prob(x).mean(0)  # H = E[-log p(x)]
    torch.testing.assert_close(monte_carlo, closed, atol=0.02, rtol=0)


def test_why_analytic_entropy_the_sample_estimate_has_zero_gradient():
    # -log p(stored x) estimates H's VALUE without bias, but x has no gradient path,
    # so its gradient is the score function, whose mean is zero: no push on entropy.
    a = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    x = Beta(a, b).sample((200_000,))
    (-Beta(a, b).log_prob(x).mean()).backward()
    grad_from_samples = a.grad.item()
    a.grad = None
    Beta(a, b).entropy().backward()
    grad_analytic = a.grad.item()
    with torch.no_grad():  # dH/dalpha = -(a - 1) psi'(a) + (a + b - 2) psi'(a + b)
        expected = (-(a - 1) * torch.polygamma(1, a) + (a + b - 2) * torch.polygamma(1, a + b)).item()
    assert abs(grad_from_samples) < 0.01  # ~0 (Monte Carlo noise ~0.001)
    assert grad_analytic == pytest.approx(expected, abs=1e-9)
    assert abs(expected) > 0.1  # a real, non-zero push (about -0.126 here)


def test_entropy_is_per_state_not_per_sample():
    actor = Actor(OBS_DIM)
    obs = torch.randn(8, OBS_DIM)
    _, ent_a, _ = actor.evaluate(obs, {MOVE: torch.full((8, 2), 0.1)})
    _, ent_b, _ = actor.evaluate(obs, {MOVE: torch.full((8, 2), 0.9)})
    torch.testing.assert_close(ent_a, ent_b)


# --- Gradients reach every parameter --------------------------------------------------------
def test_every_parameter_gets_a_gradient():
    actor, critic = Actor(OBS_DIM), Critic(14)
    obs = torch.randn(32, OBS_DIM)
    samples, _ = actor.act(obs)
    logp, entropy, _ = actor.evaluate(obs, samples)
    (-(logp * torch.randn(32)).mean() - entropy.mean()).backward()
    ((critic(torch.randn(32, 14)) - 1.0) ** 2).mean().backward()
    for net in (actor, critic):
        for name, p in net.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name


# --- Critic inputs (#1, #5) at N = 2, M = 2 ----------------------------------------------
def test_critic_inputs_and_shapes_n2_m2():
    cfg = config_from_dict({"shared": {"n_blue": 2, "n_red": 2}})
    sc = cfg.shared
    s, _ = MockEnv(sc).reset(seed=0)
    st = parse_state(s, 2, 2)
    obs, flat = ObsBuilder(sc)(st), StateFlattener(sc)(st)
    glob = critic_input("global", obs, flat)
    assert glob.shape == (2, 24) and np.array_equal(glob[0], glob[1]) and np.array_equal(glob[0], flat)
    loc = critic_input("local", obs, flat)
    assert loc.shape == (2, 29) and np.array_equal(loc, obs)
    for mode, rows in (("global", glob), ("local", loc)):
        critic = Critic(critic_input_dim(mode, obs_dim=29, state_dim=24))
        v = critic(torch.as_tensor(rows))
        assert v.shape == (2,)
        if mode == "global":
            assert v[0] == v[1]  # same s -> same V for every agent (#5)
    actor = Actor(29)
    samples, logp = actor.act(torch.as_tensor(obs))
    assert samples[MOVE].shape == (2, 2) and logp.shape == (2,)


def test_bad_critic_mode_raises():
    with pytest.raises(ValueError):
        critic_input_dim("central", 17, 14)
    with pytest.raises(ValueError):
        critic_input("central", np.zeros((1, 17)), np.zeros(14))
    with pytest.raises(ValueError):
        Actor(OBS_DIM, beta_eps=0.0)


def test_build_networks_from_config():
    cfg = config_from_dict({"rl": {"critic_input": "local", "ppo": {"hidden_sizes": [32]}, "beta_eps": 1e-5}})
    actor, critic = build_networks(cfg, obs_dim=17, state_dim=14)
    assert actor.trunk[0].in_features == 17 and actor.trunk[0].out_features == 32
    assert critic.body[0].in_features == 17  # local critic sees o_i
    assert actor.beta_eps == 1e-5
    _, critic_g = build_networks(config_from_dict({}), obs_dim=17, state_dim=14)
    assert critic_g.body[0].in_features == 14  # global critic sees flattened s


# --- Checkpoint layout and warm start (#21) -----------------------------------------------------
def test_state_dict_round_trip_and_warm_start_layout(tmp_path):
    actor = Actor(OBS_DIM)
    assert all(k.startswith(("trunk.", "heads.move.")) for k in actor.state_dict())
    path = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), path)

    torch.manual_seed(123)  # a differently initialized actor...
    fresh = Actor(OBS_DIM)
    fresh.load_state_dict(torch.load(path))  # ...becomes identical after loading
    obs = torch.randn(8, OBS_DIM)
    for name in ("concentration1", "concentration0"):
        torch.testing.assert_close(
            getattr(fresh.distributions(obs)[MOVE], name), getattr(actor.distributions(obs)[MOVE], name)
        )

    # Warm start: a future actor with an extra head loads the movement checkpoint
    # with strict=False; only the new head's parameters are missing (fresh init).
    bigger = Actor(OBS_DIM)
    bigger.heads["kick_trigger"] = nn.Linear(64, 1)  # stand-in for a later head
    result = bigger.load_state_dict(torch.load(path), strict=False)
    assert set(result.missing_keys) == {"heads.kick_trigger.weight", "heads.kick_trigger.bias"}
    assert result.unexpected_keys == []
