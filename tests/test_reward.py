# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: reward module tests (registry,
#     (1 - gamma) scaling, survival return = 1 - gamma^L, goto_target uses
#     s_next, deferred/unknown terms, effective_terminal truth table).
#   - 2026-10-14: Added the pbrs_ball_acquisition tests: the potential, every
#     boundary case (pickup, red capture, effective terminal vs truncation,
#     possession), the base/shaping split, and telescoping over mock episodes.
# Prompt summary: Implement Phase 3 step 5 of the issue #7 plan (reward module).
# Human review:   Pending review
# ==============================================================================
import dataclasses

import pytest

from marl.baselines import make_baseline
from marl.common.config import (
    BASE_REWARD_TERMS,
    SHAPING_TERMS_BUILT,
    SHAPING_TERMS_DEFERRED,
    ShapingTerm,
    config_from_dict,
)
from marl.envs.mock_env import MockEnv
from marl.rl.reward import BASE_TERMS, SHAPING_TERMS, RewardModule, ball_potential, effective_terminal

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


def s_with(blue_pos, ball_pos):
    return {
        "step": 0,
        "blue": [{"id": 0, "pos": list(blue_pos), "vel": [0.0, 0.0], "has_possession": False}],
        "red": [],
        "ball": {"pos": list(ball_pos), "vel": [0.0, 0.0]},
    }


def with_reward(cfg, **reward_changes):
    """Bypass config validation, to test the module's own checks (defense in depth)."""
    rw = dataclasses.replace(cfg.rl.reward, **reward_changes)
    return dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, reward=rw))


def test_registry_matches_config_names():
    assert set(BASE_TERMS) == BASE_REWARD_TERMS
    assert set(SHAPING_TERMS) == SHAPING_TERMS_BUILT | SHAPING_TERMS_DEFERRED


def test_survival_is_scaled_and_paid_on_terminal_step():
    cfg = config_from_dict({})
    rm = RewardModule(cfg)
    assert rm.scale == pytest.approx(0.01)
    assert rm({}, {}, {}, False, False, {}) == pytest.approx(0.01)
    assert rm({}, {}, {}, True, False, {}) == pytest.approx(0.01)


@pytest.mark.parametrize("L", [1, 10, 96, 300])
def test_survival_return_is_one_minus_gamma_to_the_L(L):
    cfg = config_from_dict({})
    rm, g = RewardModule(cfg), cfg.gamma
    ret = sum(g**t * rm({}, {}, {}, t == L - 1, False, {}) for t in range(L))
    assert ret == pytest.approx(1 - g**L)  # in [0, 1); 0.95 at L = 300 (#15)


def test_goto_target_is_negative_distance_at_s_next():
    cfg = config_from_dict(GOTO)
    rm = RewardModule(cfg)
    s = s_with((-7.0, 0.0), (3.0, 4.0))  # far away before the step...
    s_next = s_with((0.0, 0.0), (3.0, 4.0))  # ...5 m away after it
    assert rm(s, {}, s_next, False, False, {}) == pytest.approx(-5.0 * 0.01)


def test_unknown_terms_raise_value_error():
    cfg = config_from_dict({})
    with pytest.raises(ValueError):
        RewardModule(with_reward(cfg, base="win"))
    with pytest.raises(ValueError):
        RewardModule(with_reward(cfg, shaping=(ShapingTerm("touch_bonus", True, 1.0),)))


def test_enabled_deferred_term_raises_not_implemented():
    cfg = config_from_dict({})
    with pytest.raises(NotImplementedError):
        RewardModule(with_reward(cfg, shaping=(ShapingTerm("pbrs_red_distance", True, 0.1),)))


def test_disabled_deferred_term_has_no_effect():
    cfg = config_from_dict({"rl": {"reward": {"shaping": [{"name": "pbrs_red_distance", "enabled": False}]}}})
    assert RewardModule(cfg)({}, {}, {}, False, False, {}) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "terminated, truncated, mode, expected",
    [
        (False, False, "infinite", False),
        (True, False, "infinite", True),
        (False, True, "infinite", False),  # time-out -> bootstrap, not terminal
        (True, True, "infinite", True),
        (False, True, "finite", True),  # finite mode relabels time-outs as terminal
        (False, False, "finite", False),
    ],
)
def test_effective_terminal(terminated, truncated, mode, expected):
    assert effective_terminal(terminated, truncated, mode) is expected


# --- pbrs_ball_acquisition: Phi = -(closest blue to a LOOSE ball), 0 while held -----------
PBRS = "pbrs_ball_acquisition"


def ka(blue, ball, red=(), holder=None):
    """A keep-away state. ``holder`` is None (loose ball), ("blue", i) or ("red", j)."""

    def ent(team, i, pos):
        return {"id": i, "pos": list(pos), "vel": [0.0, 0.0], "has_possession": holder == (team, i)}

    return {
        "step": 0,
        "blue": [ent("blue", i, p) for i, p in enumerate(blue)],
        "red": [ent("red", j, p) for j, p in enumerate(red)],
        "ball": {"pos": list(ball), "vel": [0.0, 0.0]},
    }


def pbrs_cfg(coef=1.0, **shared):
    return config_from_dict(
        {"shared": shared, "rl": {"reward": {"shaping": [{"name": PBRS, "enabled": True, "coef": coef}]}}}
    )


def F(cfg, s, s_next, eff=False, trunc=False):
    return SHAPING_TERMS[PBRS](s, {}, s_next, eff, trunc, {}, cfg)


FAR = ka(blue=[(5.0, 0.0)], ball=(0.0, 0.0), red=[(-6.0, 0.0)])  # blue 5 m from a loose ball
CLOSER = ka(blue=[(4.9, 0.0)], ball=(0.0, 0.0), red=[(-5.91, 0.0)])  # one full-speed step later


def test_ball_potential_is_minus_the_closest_blue_distance_while_loose():
    s = ka(blue=[(3.0, 0.0), (0.0, 4.0)], ball=(0.0, 0.0), red=[(5.0, 5.0)])
    assert ball_potential(s) == pytest.approx(-3.0)  # the closest blue only (a team potential)


@pytest.mark.parametrize("holder", [("blue", 1), ("red", 0)])
def test_ball_potential_is_zero_while_anyone_holds_the_ball(holder):
    s = ka(blue=[(3.0, 0.0), (0.0, 4.0)], ball=(0.0, 4.0), red=[(5.0, 5.0)], holder=holder)
    assert ball_potential(s) == 0.0


def test_pbrs_is_gamma_phi_next_minus_phi():
    cfg = pbrs_cfg()
    g = cfg.gamma
    # = gamma * 0.1 (the approach) + (1 - gamma) * 5 (the drift): 0.099 + 0.05
    assert F(cfg, FAR, CLOSER) == pytest.approx(g * -4.9 + 5.0)
    assert F(cfg, FAR, FAR) == pytest.approx((1 - g) * 5.0)  # standing still: drift only, > 0 since Phi < 0


def test_pbrs_pickup_pays_the_last_loose_distance():
    s = ka(blue=[(0.55, 0.0)], ball=(0.0, 0.0), red=[(-6.0, 0.0)])
    picked = ka(blue=[(0.45, 0.0)], ball=(0.45, 0.0), red=[(-5.91, 0.0)], holder=("blue", 0))
    assert F(pbrs_cfg(), s, picked) == pytest.approx(0.55)  # Phi: -0.55 -> 0


def test_pbrs_red_capture_refunds_blue_distance():
    s = ka(blue=[(5.0, 0.0)], ball=(0.0, 0.0), red=[(-0.55, 0.0)])
    captured = ka(blue=[(5.0, 0.0)], ball=(-0.45, 0.0), red=[(-0.45, 0.0)], holder=("red", 0))
    assert F(pbrs_cfg(), s, captured, eff=True) == pytest.approx(5.0)  # the terminal refund: -Phi(s)


def test_pbrs_uses_zero_phi_next_only_at_an_effective_terminal():
    cfg = pbrs_cfg()
    assert F(cfg, FAR, CLOSER, eff=True) == pytest.approx(5.0)  # e.g. a finite-mode time-out, ball loose
    # an infinite-mode time-out bootstraps: the real Phi(s') is used (#15)
    assert F(cfg, FAR, CLOSER, eff=False, trunc=True) == pytest.approx(cfg.gamma * -4.9 + 5.0)


def test_pbrs_is_zero_while_blue_holds_the_ball():
    s = ka(blue=[(1.0, 1.0)], ball=(1.0, 1.0), red=[(-6.0, 0.0)], holder=("blue", 0))
    s_next = ka(blue=[(1.1, 1.0)], ball=(1.1, 1.0), red=[(-5.9, 0.0)], holder=("blue", 0))
    assert F(pbrs_cfg(), s, s_next) == 0.0  # possession starts are unshaped


@pytest.mark.parametrize("coef", [1.0, 2.5])
def test_reward_module_adds_coef_times_pbrs_and_reports_the_parts(coef):
    cfg = pbrs_cfg(coef)
    rm, g = RewardModule(cfg), cfg.gamma
    f = g * -4.9 + 5.0
    parts = rm.parts(FAR, {}, CLOSER, False, False, {})
    assert parts.base == 1.0
    assert parts.shaping == pytest.approx(coef * f)
    assert parts.total == pytest.approx((1 - g) * (1.0 + coef * f))
    assert rm(FAR, {}, CLOSER, False, False, {}) == parts.total


def test_parts_without_shaping_are_exact():
    rm = RewardModule(config_from_dict({}))
    assert rm.parts({}, {}, {}, False, False, {}) == (1.0, 0.0, rm.scale)  # bit-exact, as before shaping


@pytest.mark.parametrize("policy", ["random", "flee"])
def test_pbrs_telescopes_over_mock_episodes(policy):
    # For EVERY episode: sum_t gamma^t F_t = gamma^L * Phi(s_L) - Phi(s_0), with Phi(s_L) = 0
    # at an effective terminal. So shaping shifts every policy's return from s_0 by the same
    # amount: it can't change which policy is better (Ng et al. 1999). Free-ball starts plus a
    # short time limit cover the loose phase, pickups (flee runs for the ball), red captures,
    # and time-outs with the ball still loose. Phi(s) is read after env.step(), as in the
    # rollout, so this would also catch an env that mutates a returned state.
    cfg = pbrs_cfg(max_episode_steps=60, start={"start_mix": 0.0})
    g, term = cfg.gamma, SHAPING_TERMS[PBRS]
    env, options = MockEnv(cfg.shared), cfg.shared.start.to_reset_options()
    act = make_baseline(policy, cfg.shared, seed=0)
    seen = set()
    for seed in range(30):
        s0, _ = env.reset(seed=seed, options=options)
        s, total, t = s0, 0.0, 0
        while True:
            actions = act(s)
            s_next, terminated, truncated, info = env.step(actions)
            eff = effective_terminal(terminated, truncated, cfg.rl.horizon_mode)
            total += g**t * term(s, actions, s_next, eff, truncated, info, cfg)
            if ball_potential(s) < 0 and ball_potential(s_next) == 0 and not terminated:
                seen.add("pickup")
            s, t = s_next, t + 1
            if terminated or truncated:
                break
        seen.add("terminated" if terminated else "truncated")
        if truncated and ball_potential(s) < 0:
            seen.add("truncated_loose")
        assert total == pytest.approx(g**t * (0.0 if eff else ball_potential(s)) - ball_potential(s0), abs=1e-9)
    # Coverage: random loses races (red captures) and times out with the ball loose; flee
    # wins races (pickups) and survives to the time limit.
    assert {"random": {"terminated", "truncated_loose"}, "flee": {"pickup", "truncated"}}[policy] <= seen
