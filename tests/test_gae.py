# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: GAE tests (hand-computed boundary
#     cases, an independent reference implementation, demonstrations of the
#     merged-done and values[t+1] traps, lambda limits, input validation).
# Prompt summary: Implement Phase 3 step 7 of the issue #7 plan (buffer +
#   GAE) with hand-built boundary tests.
# Human review:   Pending review
# ==============================================================================
"""GAE tests. Hand-built cases use gamma = lambda = 0.5 (gamma*lambda = 0.25).

Common setup (T = 3, one agent): rewards [1, 1, 1], V(s_t) = 0.5 for every t.
next_value_t = V(true s_{t+1}): 0.5 for a continuing step, 0.9 for the final state
of an episode that ends at t = 1, and 0.7 for V(s_T) at the rollout end.
"""

import numpy as np
import pytest

from marl.rl.gae import compute_gae

G = LAM = 0.5
R = np.ones(3)
V = np.full((3, 1), 0.5)
F, T_ = False, True


def gae(next_values, terminated, truncated, finite=False, rewards=R, values=V, gamma=G, lam=LAM):
    term, trunc = np.array(terminated), np.array(truncated)
    eff = term | (trunc if finite else np.zeros_like(trunc))
    adv, ret = compute_gae(rewards, values, np.array(next_values, dtype=float).reshape(-1, 1), eff, term | trunc, gamma, lam)
    return adv[:, 0], ret[:, 0]


# --- Hand-computed boundary cases ----------------------------------------------------------
def test_terminated_mid_rollout():
    # d0 = 1 + .5*.5 - .5 = .75 | d1 = 1 + 0 - .5 = .5 (no bootstrap) | d2 = 1 + .5*.7 - .5 = .85
    # A2 = .85 (rollout end) | A1 = .5 (chain cut) | A0 = .75 + .25*.5 = .875
    adv, ret = gae([0.5, 0.9, 0.7], [F, T_, F], [F, F, F])
    np.testing.assert_allclose(adv, [0.875, 0.5, 0.85])
    np.testing.assert_allclose(ret, [1.375, 1.0, 1.35])  # return = A + V


def test_truncated_mid_rollout_bootstraps_from_the_true_final_state():
    # d1 = 1 + .5*.9 - .5 = .95 (bootstrap with V(final state) = .9) | A1 = .95 (chain cut)
    # A0 = .75 + .25*.95 = .9875
    adv, _ = gae([0.5, 0.9, 0.7], [F, F, F], [F, T_, F])
    np.testing.assert_allclose(adv, [0.9875, 0.95, 0.85])


def test_rollout_cut_without_episode_end():
    # d = [.75, .75, .85] | A2 = .85 | A1 = .75 + .25*.85 = .9625 | A0 = .75 + .25*.9625 = .990625
    adv, _ = gae([0.5, 0.5, 0.7], [F, F, F], [F, F, F])
    np.testing.assert_allclose(adv, [0.990625, 0.9625, 0.85])


def test_episode_terminates_exactly_at_rollout_end():
    # d2 = 1 - .5 = .5 (no bootstrap; the stored next_value .7 must be ignored)
    # A1 = .75 + .25*.5 = .875 | A0 = .75 + .25*.875 = .96875
    adv, _ = gae([0.5, 0.5, 0.7], [F, F, T_], [F, F, F])
    np.testing.assert_allclose(adv, [0.96875, 0.875, 0.5])


def test_truncation_at_rollout_end_equals_a_plain_rollout_cut():
    # Both cut the chain and bootstrap from V(true s_T): identical math. (Only LOGGING
    # differs: a rollout cut is not an episode end, #12.)
    adv_trunc, _ = gae([0.5, 0.5, 0.7], [F, F, F], [F, F, T_])
    adv_cut, _ = gae([0.5, 0.5, 0.7], [F, F, F], [F, F, F])
    np.testing.assert_allclose(adv_trunc, adv_cut)


def test_finite_mode_timeout_is_terminal():
    # horizon_mode = finite relabels a time-out as terminal: same as the terminated case.
    adv_finite, _ = gae([0.5, 0.9, 0.7], [F, F, F], [F, T_, F], finite=True)
    adv_term, _ = gae([0.5, 0.9, 0.7], [F, T_, F], [F, F, F])
    np.testing.assert_allclose(adv_finite, adv_term)


# --- Why the design is the way it is (#12) ----------------------------------------------------
def merged_done_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """CleanRL-style: done = terminated | truncated drives BOTH masks."""
    done = (np.asarray(terminated) | np.asarray(truncated)).astype(float)[:, None]
    deltas = rewards[:, None] + gamma * (1 - done) * next_values - values
    adv, running = np.zeros_like(values), np.zeros(values.shape[1])
    for t in reversed(range(len(values))):
        running = deltas[t] + gamma * lam * (1 - done[t]) * running
        adv[t] = running
    return adv[:, 0]


def test_why_two_masks_merged_done_treats_a_timeout_as_a_loss():
    nv = np.array([[0.5], [0.9], [0.7]])
    # Terminations: both versions agree.
    ours, _ = gae(nv, [F, T_, F], [F, F, F])
    np.testing.assert_allclose(merged_done_gae(R, V, nv, [F, T_, F], [F, F, F], G, LAM), ours)
    # Time-outs: the merged version drops the bootstrap (A1 = .5 instead of .95).
    ours, _ = gae(nv, [F, F, F], [F, T_, F])
    naive = merged_done_gae(R, V, nv, [F, F, F], [F, T_, F], G, LAM)
    assert ours[1] == pytest.approx(0.95) and naive[1] == pytest.approx(0.5)


def test_why_next_value_reading_values_t_plus_1_leaks_the_next_episode():
    # After a time-out at t = 1, row 2 is a NEW episode whose reset state has V = 0.1.
    values = np.array([[0.5], [0.5], [0.1]])
    adv, _ = gae([0.5, 0.9, 0.7], [F, F, F], [F, T_, F], values=values)
    assert adv[1] == pytest.approx(1 + G * 0.9 - 0.5)  # bootstraps from V(true final state) = 0.9
    leaked = 1 + G * values[2, 0] - 0.5  # what reading values[t+1] would give: V(reset state)
    assert adv[1] != pytest.approx(leaked)


def test_under_survival_reward_termination_is_the_only_loss_signal():
    # Constant reward and equal values: being caught vs. running out the clock differ
    # ONLY by the bootstrap term gamma * V(final state) (choice b, #15).
    caught, _ = gae([0.5, 0.9, 0.7], [F, T_, F], [F, F, F])
    timed_out, _ = gae([0.5, 0.9, 0.7], [F, F, F], [F, T_, F])
    assert timed_out[1] - caught[1] == pytest.approx(G * 0.9)


# --- Independent reference implementation ---------------------------------------------------
def reference_gae(rewards, values, next_values, terminated, truncated, finite, gamma, lam):
    """Definition, written forward: A_t = sum_k (gamma*lam)^k delta_{t+k}, stopping at the
    first episode end (inclusive) or the rollout end."""
    T, N = values.shape
    adv = np.zeros((T, N))
    for i in range(N):
        for t in range(T):
            total, weight = 0.0, 1.0
            for k in range(t, T):
                truly_over = terminated[k] or (finite and truncated[k])
                boot = 0.0 if truly_over else next_values[k, i]
                total += weight * (rewards[k] + gamma * boot - values[k, i])
                if terminated[k] or truncated[k]:
                    break
                weight *= gamma * lam
            adv[t, i] = total
    return adv


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("finite", [False, True])
def test_matches_reference_on_random_rollouts(seed, finite):
    rng = np.random.default_rng(seed)
    T, N = 60, 3
    rewards = rng.normal(size=T)
    values, next_values = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    terminated = rng.random(T) < 0.08
    truncated = (rng.random(T) < 0.08) & ~terminated  # the env never reports both (ICD §7)
    eff = terminated | (truncated if finite else False)
    adv, ret = compute_gae(rewards, values, next_values, eff, terminated | truncated, 0.99, 0.95)
    expected = reference_gae(rewards, values, next_values, terminated, truncated, finite, 0.99, 0.95)
    np.testing.assert_allclose(adv, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ret, expected + values, rtol=1e-12, atol=1e-12)


# --- Limits and structure ----------------------------------------------------------------------
def test_lambda_zero_gives_one_step_td_error():
    rng = np.random.default_rng(0)
    r, v, nv = rng.normal(size=10), rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
    no = np.zeros(10, bool)
    adv, _ = compute_gae(r, v, nv, no, no, 0.9, 0.0)
    np.testing.assert_allclose(adv, r[:, None] + 0.9 * nv - v)


def test_lambda_one_gives_bootstrapped_monte_carlo_return():
    rng = np.random.default_rng(1)
    T, g = 12, 0.9
    r, v, nv = rng.normal(size=T), rng.normal(size=(T, 1)), rng.normal(size=(T, 1))
    nv[:-1] = v[1:]  # one continuing episode: next_value_t = V(s_{t+1})
    no = np.zeros(T, bool)
    _, ret = compute_gae(r, v, nv, no, no, g, 1.0)
    for t in range(T):
        mc = sum(g ** (k - t) * r[k] for k in range(t, T)) + g ** (T - t) * nv[-1, 0]
        assert ret[t, 0] == pytest.approx(mc)


def test_agent_columns_are_independent_and_reward_is_shared():
    rng = np.random.default_rng(2)
    T = 20
    r, v, nv = rng.normal(size=T), rng.normal(size=(T, 2)), rng.normal(size=(T, 2))
    term = rng.random(T) < 0.1
    adv, _ = compute_gae(r, v, nv, term, term, 0.99, 0.95)
    for i in range(2):
        col, _ = compute_gae(r, v[:, i : i + 1], nv[:, i : i + 1], term, term, 0.99, 0.95)
        np.testing.assert_allclose(adv[:, i], col[:, 0])
    adv2, _ = compute_gae(np.repeat(r[:, None], 2, axis=1), v, nv, term, term, 0.99, 0.95)
    np.testing.assert_allclose(adv, adv2)  # [T] shared reward == [T, N] repeated


def test_identical_value_columns_give_identical_advantages():
    # Global critic: every agent has the same V(s) and the same team reward -> the same A_t (#5).
    rng = np.random.default_rng(3)
    v = np.repeat(rng.normal(size=(15, 1)), 3, axis=1)
    nv = np.repeat(rng.normal(size=(15, 1)), 3, axis=1)
    no = np.zeros(15, bool)
    adv, _ = compute_gae(rng.normal(size=15), v, nv, no, no, 0.99, 0.95)
    assert np.all(adv == adv[:, :1])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"effective_terminal": np.array([F, T_, F]), "episode_end": np.array([F, F, F])}, "imply"),
        ({"values": np.zeros(3)}, r"\[T, N\]"),
        ({"next_values": np.zeros((3, 2))}, "shape mismatch"),
        ({"rewards": np.zeros(4)}, "shape mismatch"),
        ({"episode_end": np.zeros(4, bool)}, "masks"),
        ({"gamma": 1.5}, "gamma"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    args = {
        "rewards": R,
        "values": V,
        "next_values": np.full((3, 1), 0.5),
        "effective_terminal": np.zeros(3, bool),
        "episode_end": np.zeros(3, bool),
        "gamma": G,
        "lam": LAM,
    } | kwargs
    with pytest.raises(ValueError, match=match):
        compute_gae(**args)
